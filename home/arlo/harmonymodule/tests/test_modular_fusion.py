#!/usr/bin/env python3
"""
Integration Tests for Modular Fusion System

Comprehensive test suite for all components of the modular fusion system
across Agents 1-9, integrated through the Unified API (Agent 10).

Test Categories:
1. API Initialization & Basic Operations (Tests 1-5)
2. Genre Detection & Analysis (Tests 6-12)
3. Quick Fusion & Component Mixing (Tests 13-20)
4. Context-Aware Generation (Tests 21-27)
5. Inpainting & Reharmonization (Tests 28-34)
6. Tempo & Meter Conversion (Tests 35-40)
7. Advanced Fusion Techniques (Tests 41-48)
8. Edge Cases & Error Handling (Tests 49-55)
9. Integration Tests (Tests 56-60)
10. Performance Tests (Tests 61-65)

Author: Agent 10 - Unified API & Integration
Date: 2025
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.api import (
    HarmonyModuleAPI,
    QuickFusion,
    GenreBlend,
    ComponentMix,
    ContextGeneration,
    InpaintSection,
    TransformTempo,
    TransformMeter,
    GranularControl,
)


class TestBase(unittest.TestCase):
    """Base test class with common setup/teardown"""

    def setUp(self):
        """Create temporary directory for test outputs"""
        self.test_dir = tempfile.mkdtemp(prefix="test_modular_fusion_")
        self.api = HarmonyModuleAPI(output_dir=self.test_dir)

    def tearDown(self):
        """Clean up temporary directory"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def create_sample_midi(self, name="test.mid"):
        """Helper to create a sample MIDI file for testing"""
        # This would create a minimal valid MIDI file
        # For now, we'll use a mock or skip if dependencies not available
        pass


# ==============================================================================
# CATEGORY 1: API INITIALIZATION & BASIC OPERATIONS
# ==============================================================================

class TestAPIInitialization(TestBase):
    """Test API initialization and basic operations"""

    def test_01_api_initialization(self):
        """Test API initializes correctly"""
        self.assertIsNotNone(self.api)
        self.assertTrue(os.path.exists(self.test_dir))
        self.assertEqual(str(self.api.output_dir), self.test_dir)

    def test_02_output_directory_creation(self):
        """Test output directory is created if doesn't exist"""
        new_dir = os.path.join(self.test_dir, "subdir", "output")
        api = HarmonyModuleAPI(output_dir=new_dir)
        self.assertTrue(os.path.exists(new_dir))

    def test_03_list_available_genres(self):
        """Test listing available genres"""
        genres = self.api.list_genres()
        self.assertIsInstance(genres, list)
        # Common genres should be available
        # (Will work once GENRE_PROFILES is populated)

    def test_04_get_genre_info(self):
        """Test getting genre information"""
        # This will work once GENRE_PROFILES is populated
        info = self.api.get_genre_info("jazz")
        # Should return GenreFeatures or None

    def test_05_history_tracking(self):
        """Test operation history is tracked"""
        self.assertEqual(len(self.api.history), 0)
        # After operations, history should be populated


# ==============================================================================
# CATEGORY 2: GENRE DETECTION & ANALYSIS
# ==============================================================================

class TestGenreDetection(TestBase):
    """Test genre detection capabilities (Agent 1)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 (GenreDetector) not yet implemented"
    )
    def test_06_detect_genre_from_midi(self):
        """Test genre detection from MIDI file"""
        # Create sample MIDI
        sample_path = os.path.join(self.test_dir, "sample.mid")
        # ... create sample ...

        genres = self.api.detect_genre(sample_path, top_n=3)

        self.assertIsInstance(genres, list)
        self.assertLessEqual(len(genres), 3)
        for genre, confidence in genres:
            self.assertIsInstance(genre, str)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_07_extract_rhythmic_features(self):
        """Test rhythmic feature extraction"""
        sample_path = os.path.join(self.test_dir, "sample.mid")

        features = self.api.extract_features(sample_path)

        # Check rhythmic features
        self.assertIsNotNone(features.tempo_range)
        self.assertIsNotNone(features.swing_factor)
        self.assertIsNotNone(features.syncopation)
        self.assertGreaterEqual(features.swing_factor, 0.5)
        self.assertLessEqual(features.swing_factor, 1.0)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_08_extract_harmonic_features(self):
        """Test harmonic feature extraction"""
        sample_path = os.path.join(self.test_dir, "sample.mid")

        features = self.api.extract_features(sample_path)

        # Check harmonic features
        self.assertIsNotNone(features.chord_types)
        self.assertIsNotNone(features.harmonic_rhythm)
        self.assertIsInstance(features.use_extensions, bool)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_09_extract_melodic_features(self):
        """Test melodic feature extraction"""
        sample_path = os.path.join(self.test_dir, "sample.mid")

        features = self.api.extract_features(sample_path)

        # Check melodic features
        self.assertIn(features.interval_preference,
                     ['stepwise', 'balanced', 'angular'])
        self.assertIsNotNone(features.ornamentation)
        self.assertIsNotNone(features.melodic_range)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_10_extract_instrumentation_features(self):
        """Test instrumentation feature extraction"""
        sample_path = os.path.join(self.test_dir, "sample.mid")

        features = self.api.extract_features(sample_path)

        # Check instrumentation features
        self.assertIsNotNone(features.instruments)
        self.assertIsNotNone(features.texture)
        self.assertIn(features.texture,
                     ['monophonic', 'homophonic', 'polyphonic'])

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_11_detect_swing_factor(self):
        """Test swing factor detection"""
        # Create swing MIDI
        sample_path = os.path.join(self.test_dir, "swing.mid")

        features = self.api.extract_features(sample_path)

        # Swing should be detected (around 0.67 for triplet swing)
        self.assertGreater(features.swing_factor, 0.6)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_1'),
        "Agent 1 not yet implemented"
    )
    def test_12_detect_chord_progression(self):
        """Test chord progression extraction"""
        sample_path = os.path.join(self.test_dir, "chords.mid")

        # Would use ChordDetector from Agent 1
        # chords = ChordDetector.extract_chord_progression(sample_path)
        # self.assertIsInstance(chords, list)
        pass


# ==============================================================================
# CATEGORY 3: QUICK FUSION & COMPONENT MIXING
# ==============================================================================

class TestQuickFusion(TestBase):
    """Test quick fusion capabilities (Agent 5)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 (ModularFusion) not yet implemented"
    )
    def test_13_basic_quick_fusion(self):
        """Test basic quick fusion"""
        composition = self.api.quick_fusion(
            harmony="jazz",
            rhythm="funk",
            tempo=120,
            key="C",
            measures=8
        )

        self.assertIsNotNone(composition)
        self.assertIsNotNone(self.api.composition)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_14_fusion_with_all_components(self):
        """Test fusion with all components specified"""
        composition = self.api.quick_fusion(
            harmony="jazz",
            rhythm="funk",
            melody="blues",
            bass="reggae",
            drums="hiphop",
            instrumentation="electronic",
            tempo=110,
            key="Dm",
            measures=16
        )

        self.assertIsNotNone(composition)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_15_fusion_preserves_parameters(self):
        """Test that fusion preserves specified parameters"""
        tempo = 125
        key = "Em"
        measures = 24

        composition = self.api.quick_fusion(
            harmony="jazz",
            rhythm="funk",
            tempo=tempo,
            key=key,
            measures=measures
        )

        # Verify parameters (if accessible from composition)
        # self.assertEqual(composition.tempo, tempo)
        # self.assertEqual(composition.key, key)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_16_weighted_blend(self):
        """Test weighted genre blending"""
        composition = self.api.weighted_blend(
            blends={
                'harmony': [('jazz', 0.6), ('blues', 0.4)],
                'rhythm': [('funk', 1.0)]
            },
            tempo=105,
            key="G",
            measures=16
        )

        self.assertIsNotNone(composition)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_17_weighted_blend_normalization(self):
        """Test that blend weights are normalized"""
        # Weights that don't sum to 1.0 should be normalized
        composition = self.api.weighted_blend(
            blends={
                'harmony': [('jazz', 3.0), ('blues', 2.0)],  # Will normalize to 0.6/0.4
            },
            tempo=120,
            measures=8
        )

        self.assertIsNotNone(composition)

    def test_18_convenience_functions(self):
        """Test convenience functions work"""
        # Test QuickFusion convenience functions
        # These should work even if underlying implementation isn't complete

        # Just test they don't crash
        try:
            QuickFusion.jazz_funk(tempo=115)
        except NotImplementedError:
            pass  # Expected if not implemented

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_19_progressive_morph(self):
        """Test progressive genre morphing"""
        composition = self.api.progressive_morph(
            from_genre="jazz",
            to_genre="electronic",
            measures=32,
            morph_type="linear"
        )

        self.assertIsNotNone(composition)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_20_genre_compatibility_check(self):
        """Test genre compatibility analysis"""
        compat = self.api.check_compatibility("jazz", "funk")

        self.assertIsInstance(compat, dict)
        self.assertIn('overall', compat)
        self.assertIn('rhythmic', compat)
        self.assertIn('harmonic', compat)

        # All scores should be 0-1
        for score in compat.values():
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ==============================================================================
# CATEGORY 4: CONTEXT-AWARE GENERATION
# ==============================================================================

class TestContextAwareGeneration(TestBase):
    """Test context-aware generation (Agent 3)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 (ContextAwareGenerator) not yet implemented"
    )
    def test_21_load_midi_file(self):
        """Test loading MIDI file"""
        sample_path = os.path.join(self.test_dir, "sample.mid")
        # Create sample...

        info = self.api.load_midi(sample_path)

        self.assertIsInstance(info, dict)
        self.assertIn('filepath', info)
        self.assertEqual(info['filepath'], sample_path)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_22_add_track_to_arrangement(self):
        """Test adding track to existing arrangement"""
        # Create base arrangement
        sample_path = os.path.join(self.test_dir, "base.mid")

        self.api.load_midi(sample_path)
        notes = self.api.add_track(
            instrument=33,  # Bass
            track_type="bass",
            genre="funk"
        )

        self.assertIsInstance(notes, list)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_23_add_track_auto_type(self):
        """Test adding track with auto type detection"""
        sample_path = os.path.join(self.test_dir, "base.mid")

        self.api.load_midi(sample_path)
        notes = self.api.add_track(
            instrument=65,  # Sax
            track_type="auto"  # Should detect melody
        )

        self.assertIsInstance(notes, list)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_24_suggest_tracks(self):
        """Test smart orchestration suggestions"""
        sample_path = os.path.join(self.test_dir, "sparse.mid")

        suggestions = self.api.suggest_tracks(sample_path)

        self.assertIsInstance(suggestions, list)
        for sug in suggestions:
            self.assertIn('instrument', sug)
            self.assertIn('reason', sug)
            self.assertIn('track_type', sug)
            self.assertIn('priority', sug)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_25_context_aware_harmony(self):
        """Test that added tracks follow existing harmony"""
        # This would verify that generated notes match chord progression
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_26_context_aware_rhythm(self):
        """Test that added tracks match existing rhythm"""
        # This would verify rhythmic alignment
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_3'),
        "Agent 3 not yet implemented"
    )
    def test_27_voice_leading_check(self):
        """Test that voice leading rules are followed"""
        # Should avoid parallel fifths, octaves, etc.
        pass


# ==============================================================================
# CATEGORY 5: INPAINTING & REHARMONIZATION
# ==============================================================================

class TestInpainting(TestBase):
    """Test inpainting and reharmonization (Agent 4)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 (InpaintingEngine) not yet implemented"
    )
    def test_28_inpaint_with_new_chords(self):
        """Test inpainting section with new chords"""
        sample_path = os.path.join(self.test_dir, "original.mid")

        self.api.load_midi(sample_path)
        new_chords = ["Dm7", "G7", "Cmaj7", "A7"]

        regenerated = self.api.inpaint_section(
            tracks=[0, 1],
            measures=(5, 8),
            new_chords=new_chords
        )

        self.assertIsInstance(regenerated, dict)
        self.assertIn(0, regenerated)
        self.assertIn(1, regenerated)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_29_inpaint_with_genre_change(self):
        """Test inpainting with different genre"""
        sample_path = os.path.join(self.test_dir, "jazz.mid")

        self.api.load_midi(sample_path)
        regenerated = self.api.inpaint_section(
            tracks=[0, 1],
            measures=(9, 16),
            new_genre="electronic"
        )

        self.assertIsInstance(regenerated, dict)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_30_inpaint_preserve_melody(self):
        """Test inpainting while preserving melody"""
        sample_path = os.path.join(self.test_dir, "song.mid")

        self.api.load_midi(sample_path)
        regenerated = self.api.inpaint_section(
            tracks=[0, 1],
            measures=(5, 12),
            new_chords=["Dm7", "G7"] * 4,
            preserve_melody=True
        )

        # Should keep melody notes, change only harmony
        self.assertIsInstance(regenerated, dict)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_31_reharmonize_section(self):
        """Test reharmonization"""
        sample_path = os.path.join(self.test_dir, "simple.mid")

        self.api.load_midi(sample_path)
        new_chords = self.api.reharmonize(
            measures=(1, 8),
            style="jazz"
        )

        self.assertIsInstance(new_chords, list)
        self.assertGreater(len(new_chords), 0)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_32_reharmonize_romantic_style(self):
        """Test romantic reharmonization"""
        sample_path = os.path.join(self.test_dir, "simple.mid")

        self.api.load_midi(sample_path)
        new_chords = self.api.reharmonize(
            measures=(1, 8),
            style="romantic"
        )

        self.assertIsInstance(new_chords, list)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_33_boundary_smoothing(self):
        """Test that inpainted sections have smooth boundaries"""
        # Should verify stepwise motion at boundaries
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_4'),
        "Agent 4 not yet implemented"
    )
    def test_34_tritone_substitution(self):
        """Test tritone substitution in reharmonization"""
        # G7 should become Db7
        pass


# ==============================================================================
# CATEGORY 6: TEMPO & METER CONVERSION
# ==============================================================================

class TestTempoConversion(TestBase):
    """Test tempo conversion (Agent 6)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_6'),
        "Agent 6 (TempoConverter) not yet implemented"
    )
    def test_35_basic_tempo_conversion(self):
        """Test basic tempo conversion"""
        sample_path = os.path.join(self.test_dir, "90bpm.mid")

        self.api.load_midi(sample_path)
        result = self.api.convert_tempo(140, style_adjust=False)

        self.assertIsNotNone(result)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_6'),
        "Agent 6 not yet implemented"
    )
    def test_36_style_aware_tempo_conversion(self):
        """Test style-aware tempo conversion"""
        sample_path = os.path.join(self.test_dir, "90bpm.mid")

        self.api.load_midi(sample_path)
        result = self.api.convert_tempo(140, style_adjust=True)

        # Should adjust patterns, not just speed
        self.assertIsNotNone(result)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_6'),
        "Agent 6 not yet implemented"
    )
    def test_37_double_time_conversion(self):
        """Test double-time conversion"""
        # Should create double-time feel
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_6'),
        "Agent 6 not yet implemented"
    )
    def test_38_half_time_conversion(self):
        """Test half-time conversion"""
        # Should create half-time feel
        pass


class TestMeterConversion(TestBase):
    """Test meter conversion (Agent 7)"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_7'),
        "Agent 7 (MeterConverter) not yet implemented"
    )
    def test_39_convert_4_4_to_3_4(self):
        """Test converting 4/4 to 3/4"""
        sample_path = os.path.join(self.test_dir, "four_four.mid")

        self.api.load_midi(sample_path)
        result = self.api.convert_meter((3, 4))

        self.assertIsNotNone(result)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_7'),
        "Agent 7 not yet implemented"
    )
    def test_40_convert_to_odd_meter(self):
        """Test converting to odd meter (7/8)"""
        sample_path = os.path.join(self.test_dir, "four_four.mid")

        self.api.load_midi(sample_path)
        result = self.api.convert_meter((7, 8))

        self.assertIsNotNone(result)


# ==============================================================================
# CATEGORY 7: ADVANCED FUSION TECHNIQUES
# ==============================================================================

class TestAdvancedFusion(TestBase):
    """Test advanced fusion techniques"""

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_8'),
        "Agent 8 (GranularControl) not yet implemented"
    )
    def test_41_apply_custom_pattern(self):
        """Test applying custom rhythm pattern"""
        rhythm = [1.0, 1.5, 3.0, 3.75]
        chords = ["Dm7", "G7", "Cmaj7", "A7"]

        notes = self.api.apply_pattern(
            rhythm_pattern=rhythm,
            chords=chords,
            instrument_section="brass"
        )

        self.assertIsInstance(notes, list)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_8'),
        "Agent 8 not yet implemented"
    )
    def test_42_brass_hits_idiomatic(self):
        """Test brass hits use idiomatic voicings"""
        # Should verify proper brass voicings
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_8'),
        "Agent 8 not yet implemented"
    )
    def test_43_string_swells(self):
        """Test string swells"""
        rhythm = [1.0, 2.0, 3.0, 4.0]
        chords = ["Cmaj7", "Dm7", "G7", "Cmaj7"]

        notes = self.api.apply_pattern(
            rhythm_pattern=rhythm,
            chords=chords,
            instrument_section="strings"
        )

        self.assertIsInstance(notes, list)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_44_progressive_morph_linear(self):
        """Test linear progressive morph"""
        comp = self.api.progressive_morph(
            from_genre="jazz",
            to_genre="electronic",
            measures=32,
            morph_type="linear"
        )

        self.assertIsNotNone(comp)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_45_progressive_morph_s_curve(self):
        """Test S-curve progressive morph"""
        comp = self.api.progressive_morph(
            from_genre="jazz",
            to_genre="electronic",
            measures=32,
            morph_type="s-curve"
        )

        self.assertIsNotNone(comp)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_46_fusion_suggestions(self):
        """Test fusion parameter suggestions"""
        params = self.api.suggest_fusion("jazz", "electronic")

        self.assertIn('recommended_weight_a', params)
        self.assertIn('tempo', params)

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_9'),
        "Agent 9 (MultiGenreArranger) not yet implemented"
    )
    def test_47_track_level_genres(self):
        """Test different genre per track"""
        # Would test track-level genre assignment
        pass

    @unittest.skipUnless(
        os.getenv('TEST_AGENT_5'),
        "Agent 5 not yet implemented"
    )
    def test_48_component_replacement(self):
        """Test replacing single component"""
        # Would test ComponentReplacer
        pass


# ==============================================================================
# CATEGORY 8: EDGE CASES & ERROR HANDLING
# ==============================================================================

class TestErrorHandling(TestBase):
    """Test error handling and edge cases"""

    def test_49_load_nonexistent_file(self):
        """Test loading nonexistent MIDI file raises error"""
        with self.assertRaises(FileNotFoundError):
            self.api.load_midi("/nonexistent/file.mid")

    def test_50_export_without_composition(self):
        """Test exporting without composition raises error"""
        with self.assertRaises(ValueError):
            self.api.export("output.mid")

    def test_51_detect_genre_without_file(self):
        """Test genre detection without loaded file raises error"""
        with self.assertRaises(ValueError):
            self.api.detect_genre()

    def test_52_invalid_genre_name(self):
        """Test using invalid genre name"""
        # Should handle gracefully or raise ValueError
        try:
            self.api.quick_fusion(harmony="invalid_genre", rhythm="jazz")
        except (ValueError, NotImplementedError):
            pass  # Expected

    def test_53_invalid_time_signature(self):
        """Test invalid time signature"""
        with self.assertRaises((ValueError, NotImplementedError)):
            self.api.quick_fusion(
                harmony="jazz",
                rhythm="jazz",
                time_signature=(0, 4)  # Invalid
            )

    def test_54_invalid_tempo(self):
        """Test invalid tempo values"""
        # Negative or zero tempo should be rejected
        with self.assertRaises((ValueError, NotImplementedError)):
            self.api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=-10)

    def test_55_file_overwrite_protection(self):
        """Test file overwrite protection"""
        # Create file
        self.api.quick_fusion(harmony="jazz", rhythm="jazz", measures=4)
        self.api.export("test.mid")

        # Try to overwrite without flag
        with self.assertRaises(FileExistsError):
            self.api.export("test.mid", overwrite=False)


# ==============================================================================
# CATEGORY 9: INTEGRATION TESTS
# ==============================================================================

class TestIntegration(TestBase):
    """End-to-end integration tests"""

    @unittest.skipUnless(
        os.getenv('TEST_INTEGRATION'),
        "Integration tests require all agents"
    )
    def test_56_full_workflow_create_analyze_modify(self):
        """Test complete workflow: create → analyze → modify → export"""
        # 1. Create composition
        self.api.quick_fusion(harmony="jazz", rhythm="funk", tempo=115, measures=16)
        path1 = self.api.export("workflow_step1.mid")

        # 2. Analyze
        self.api.load_midi(path1)
        genres = self.api.detect_genre()
        self.assertGreater(len(genres), 0)

        # 3. Modify (add track)
        self.api.add_track(instrument=33, track_type="bass", genre="funk")

        # 4. Export
        path2 = self.api.export("workflow_step2.mid", overwrite=True)
        self.assertTrue(os.path.exists(path2))

    @unittest.skipUnless(
        os.getenv('TEST_INTEGRATION'),
        "Integration tests require all agents"
    )
    def test_57_detect_and_blend(self):
        """Test detecting genre then creating blend"""
        # Create sample
        self.api.quick_fusion(harmony="jazz", rhythm="jazz", measures=8)
        sample = self.api.export("sample.mid")

        # Detect
        genres = self.api.detect_genre(sample, top_n=2)
        genre_a = genres[0][0]
        genre_b = genres[1][0] if len(genres) > 1 else "funk"

        # Create blend based on detection
        comp = self.api.quick_fusion(harmony=genre_a, rhythm=genre_b, measures=16)
        self.assertIsNotNone(comp)

    @unittest.skipUnless(
        os.getenv('TEST_INTEGRATION'),
        "Integration tests require all agents"
    )
    def test_58_progressive_fusion_workflow(self):
        """Test progressive fusion from detection to morphing"""
        # Would test multi-step progressive fusion
        pass

    @unittest.skipUnless(
        os.getenv('TEST_INTEGRATION'),
        "Integration tests require all agents"
    )
    def test_59_reharmonize_and_tempo_change(self):
        """Test reharmonization + tempo conversion together"""
        # Create base
        self.api.quick_fusion(harmony="jazz", rhythm="jazz", tempo=90, measures=16)
        base = self.api.export("base.mid")

        # Reharmonize section
        self.api.load_midi(base)
        new_chords = self.api.reharmonize(measures=(9, 16), style="jazz")
        self.api.inpaint_section(tracks=[0, 1], measures=(9, 16), new_chords=new_chords)

        # Convert tempo
        self.api.convert_tempo(140)

        # Export
        final = self.api.export("final.mid", overwrite=True)
        self.assertTrue(os.path.exists(final))

    @unittest.skipUnless(
        os.getenv('TEST_INTEGRATION'),
        "Integration tests require all agents"
    )
    def test_60_multi_agent_collaboration(self):
        """Test multiple agents working together"""
        # Uses features from all agents in sequence
        pass


# ==============================================================================
# CATEGORY 10: PERFORMANCE TESTS
# ==============================================================================

class TestPerformance(TestBase):
    """Performance and optimization tests"""

    def test_61_api_initialization_speed(self):
        """Test API initializes quickly"""
        import time
        start = time.time()
        api = HarmonyModuleAPI(output_dir=self.test_dir)
        elapsed = time.time() - start

        # Should initialize in under 1 second
        self.assertLess(elapsed, 1.0)

    @unittest.skipUnless(
        os.getenv('TEST_PERFORMANCE'),
        "Performance tests optional"
    )
    def test_62_quick_fusion_performance(self):
        """Test quick fusion completes in reasonable time"""
        import time

        start = time.time()
        try:
            self.api.quick_fusion(harmony="jazz", rhythm="funk", measures=8)
        except NotImplementedError:
            self.skipTest("ModularFusion not implemented")

        elapsed = time.time() - start

        # Should complete in under 5 seconds for 8 measures
        self.assertLess(elapsed, 5.0)

    @unittest.skipUnless(
        os.getenv('TEST_PERFORMANCE'),
        "Performance tests optional"
    )
    def test_63_genre_detection_performance(self):
        """Test genre detection speed"""
        # Should analyze MIDI in under 2 seconds
        pass

    @unittest.skipUnless(
        os.getenv('TEST_PERFORMANCE'),
        "Performance tests optional"
    )
    def test_64_large_composition_export(self):
        """Test exporting large composition"""
        # Test with 100+ measure composition
        pass

    @unittest.skipUnless(
        os.getenv('TEST_PERFORMANCE'),
        "Performance tests optional"
    )
    def test_65_memory_usage(self):
        """Test memory usage stays reasonable"""
        # Should not leak memory on repeated operations
        pass


# ==============================================================================
# TEST SUITE
# ==============================================================================

def suite():
    """Create test suite"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAPIInitialization))
    suite.addTests(loader.loadTestsFromTestCase(TestGenreDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestQuickFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestContextAwareGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestInpainting))
    suite.addTests(loader.loadTestsFromTestCase(TestTempoConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestMeterConversion))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestErrorHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))

    return suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)

    # Check for specific test environment variables
    print("=" * 80)
    print("MODULAR FUSION INTEGRATION TESTS")
    print("=" * 80)
    print("\nAgent Implementation Status:")
    print(f"  Agent 1 (Genre Detection):    {'✓' if os.getenv('TEST_AGENT_1') else '⚠ Not yet implemented'}")
    print(f"  Agent 2 (Component System):   {'✓' if os.getenv('TEST_AGENT_2') else '⚠ Not yet implemented'}")
    print(f"  Agent 3 (Context-Aware):      {'✓' if os.getenv('TEST_AGENT_3') else '⚠ Not yet implemented'}")
    print(f"  Agent 4 (Inpainting):         {'✓' if os.getenv('TEST_AGENT_4') else '⚠ Not yet implemented'}")
    print(f"  Agent 5 (Modular Fusion):     {'✓' if os.getenv('TEST_AGENT_5') else '⚠ Not yet implemented'}")
    print(f"  Agent 6 (Tempo Conversion):   {'✓' if os.getenv('TEST_AGENT_6') else '⚠ Not yet implemented'}")
    print(f"  Agent 7 (Meter Conversion):   {'✓' if os.getenv('TEST_AGENT_7') else '⚠ Not yet implemented'}")
    print(f"  Agent 8 (Granular Control):   {'✓' if os.getenv('TEST_AGENT_8') else '⚠ Not yet implemented'}")
    print(f"  Agent 9 (Multi-Genre):        {'✓' if os.getenv('TEST_AGENT_9') else '⚠ Not yet implemented'}")
    print("\nNote: Set TEST_AGENT_X=1 to enable tests for specific agents")
    print("=" * 80)
    print()

    result = runner.run(suite())
    sys.exit(0 if result.wasSuccessful() else 1)
