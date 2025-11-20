"""
Test Suite for Hierarchical Parameter System - Agent 01 Phase 2
===============================================================

Comprehensive tests for the hierarchical parameter extraction,
validation, and backward compatibility.

Author: Agent 01 - Parameter Consolidation Architect
Date: November 20, 2025
Version: 2.0.0
"""

import unittest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from midi_generator.parameters.hierarchical_extractor import HierarchicalParameterExtractor
from midi_generator.parameters.legacy_adapter import LegacyParameterAdapter
from midi_generator.parameters.hierarchical_validator import HierarchicalParameterValidator


class TestHierarchicalExtractor(unittest.TestCase):
    """Test hierarchical parameter extraction"""

    def setUp(self):
        self.extractor = HierarchicalParameterExtractor(verbose=False)

    def test_key_detection(self):
        """Test key detection algorithm"""
        # C major pitches
        c_major_pitches = [60, 62, 64, 65, 67, 69, 71, 72]  # C D E F G A B C
        tonic, mode = self.extractor._detect_key(c_major_pitches)
        self.assertEqual(tonic, 'C')
        self.assertEqual(mode, 'major')

    def test_energy_computation(self):
        """Test energy level computation"""
        energy = self.extractor._compute_energy_level(
            tempo=180.0,
            dynamics=0.8,
            density=8.0
        )
        self.assertGreater(energy, 0.0)
        self.assertLessEqual(energy, 1.0)

        # High tempo + high dynamics = high energy
        high_energy = self.extractor._compute_energy_level(
            tempo=200.0,
            dynamics=0.9,
            density=10.0
        )
        self.assertGreater(high_energy, 0.7)

    def test_swing_categorization(self):
        """Test swing amount categorization"""
        self.assertEqual(self.extractor._categorize_swing(0.50), 'straight')
        self.assertEqual(self.extractor._categorize_swing(0.60), 'light')
        self.assertEqual(self.extractor._categorize_swing(0.67), 'medium')
        self.assertEqual(self.extractor._categorize_swing(0.75), 'hard')


class TestLegacyAdapter(unittest.TestCase):
    """Test backward compatibility adapter"""

    def setUp(self):
        self.adapter = LegacyParameterAdapter(show_deprecation_warnings=False)

    def test_old_to_new_direct_mapping(self):
        """Test direct parameter mapping"""
        old_params = {
            'rhythm.swing.amount': 0.67,
            'rhythm.syncopation.probability': 0.4,
            'harmony.voicing.spread': 0.6
        }

        new_params = self.adapter.old_to_new(old_params)

        # Check direct mappings
        self.assertEqual(
            new_params['level2_universal']['rhythm']['swing_amount'],
            0.67
        )
        self.assertEqual(
            new_params['level2_universal']['rhythm']['syncopation'],
            0.4
        )
        self.assertEqual(
            new_params['level2_universal']['harmony']['voicing_spread'],
            0.6
        )

    def test_old_to_new_merged_mapping(self):
        """Test merged parameter mapping"""
        old_params = {
            'harmony.extensions.use_9ths': True,
            'harmony.extensions.use_11ths': True,
            'harmony.extensions.use_13ths': False
        }

        new_params = self.adapter.old_to_new(old_params)

        # Check merged mapping (formula: 0.3*9ths + 0.3*11ths + 0.4*13ths)
        expected_complexity = 0.3 * 1.0 + 0.3 * 1.0 + 0.4 * 0.0
        self.assertAlmostEqual(
            new_params['level2_universal']['harmony']['complexity'],
            expected_complexity,
            places=2
        )

    def test_new_to_old_reverse_mapping(self):
        """Test reverse mapping (new → old)"""
        new_params = {
            'level1_global': {
                'genre.primary': 'jazz',
                'tempo.bpm': 180.0
            },
            'level2_universal': {
                'rhythm': {
                    'swing_amount': 0.67,
                    'syncopation': 0.5
                },
                'harmony': {
                    'voicing_spread': 0.7
                }
            },
            'level3_genre_specific': {}
        }

        old_params = self.adapter.new_to_old(new_params)

        # Check reverse mappings
        self.assertEqual(old_params['rhythm.swing.amount'], 0.67)
        self.assertEqual(old_params['rhythm.syncopation.probability'], 0.5)
        self.assertEqual(old_params['harmony.voicing.spread'], 0.7)

    def test_conversion_preservation(self):
        """Test that old→new→old preserves important values"""
        old_params = {
            'rhythm.swing.amount': 0.67,
            'rhythm.syncopation.probability': 0.4,
            'harmony.voicing.spread': 0.6,
            'dynamics.velocity.base': 90
        }

        # Convert old → new → old
        new_params = self.adapter.old_to_new(old_params)
        reconstructed = self.adapter.new_to_old(new_params)

        # Check preservation
        self.assertEqual(reconstructed['rhythm.swing.amount'], 0.67)
        self.assertEqual(reconstructed['rhythm.syncopation.probability'], 0.4)
        self.assertEqual(reconstructed['harmony.voicing.spread'], 0.6)
        # Dynamics may have some rounding
        self.assertAlmostEqual(reconstructed['dynamics.velocity.base'], 90, delta=2)


class TestHierarchicalValidator(unittest.TestCase):
    """Test parameter validation"""

    def setUp(self):
        self.validator = HierarchicalParameterValidator()

    def test_valid_parameters(self):
        """Test validation of valid parameters"""
        valid_params = {
            'level1_global': {
                'genre.primary': 'jazz',
                'tempo.bpm': 120.0,
                'time_signature': '4/4',
                'key.tonic': 'C',
                'key.mode': 'major',
                'energy.level': 0.6,
                'complexity.overall': 0.5,
                'structure.form': 'AABA'
            },
            'level2_universal': {
                'harmony': {'complexity': 0.7},
                'melody': {'range_semitones': 12},
                'rhythm': {'subdivision': 'eighth'},
                'dynamics': {'overall_level': 0.6},
                'texture': {'polyphony': 4}
            },
            'level3_genre_specific': {
                'orchestration': {'instrument_count': 5}
            }
        }

        result = self.validator.validate_all(valid_params)
        self.assertTrue(result.is_valid)
        self.assertEqual(len(result.errors), 0)

    def test_invalid_ranges(self):
        """Test detection of out-of-range values"""
        invalid_params = {
            'level1_global': {
                'energy.level': 1.5,  # Out of range
                'tempo.bpm': 300.0    # Warning
            },
            'level2_universal': {
                'harmony': {'complexity': 2.0},  # Out of range
                'dynamics': {},
                'melody': {},
                'rhythm': {},
                'texture': {}
            },
            'level3_genre_specific': {}
        }

        result = self.validator.validate_all(invalid_params)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_invalid_types(self):
        """Test detection of invalid types"""
        invalid_params = {
            'level1_global': {
                'genre.primary': 'invalid_genre',  # Invalid option
                'energy.level': 0.5
            },
            'level2_universal': {
                'rhythm': {'subdivision': 'invalid'},  # Invalid option
                'harmony': {},
                'melody': {},
                'dynamics': {},
                'texture': {}
            },
            'level3_genre_specific': {}
        }

        result = self.validator.validate_all(invalid_params)
        self.assertFalse(result.is_valid)
        self.assertGreater(len(result.errors), 0)

    def test_cross_parameter_validation(self):
        """Test cross-parameter consistency checks"""
        params = {
            'level1_global': {
                'genre.primary': 'jazz',
                'energy.level': 0.9,  # High energy
                'complexity.overall': 0.7
            },
            'level2_universal': {
                'dynamics': {'overall_level': 0.2},  # But low dynamics (inconsistent)
                'harmony': {'complexity': 0.7},
                'melody': {},
                'rhythm': {},
                'texture': {}
            },
            'level3_genre_specific': {}
        }

        result = self.validator.validate_all(params)
        # Should have warnings about inconsistency
        self.assertGreater(len(result.warnings), 0)

    def test_genre_parameter_matching(self):
        """Test that genre-specific params match genre"""
        params = {
            'level1_global': {
                'genre.primary': 'rock'  # Genre is rock
            },
            'level2_universal': {
                'harmony': {},
                'melody': {},
                'rhythm': {},
                'dynamics': {},
                'texture': {}
            },
            'level3_genre_specific': {
                'jazz': {  # But jazz params are present (warning)
                    'walking_bass': 0.8
                }
            }
        }

        result = self.validator.validate_all(params)
        # Should have warning about genre mismatch
        self.assertGreater(len(result.warnings), 0)


def run_tests():
    """Run all tests"""
    print("="*80)
    print("HIERARCHICAL PARAMETER SYSTEM - TEST SUITE")
    print("="*80)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED")
    else:
        print("\n❌ SOME TESTS FAILED")

    print("="*80)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
