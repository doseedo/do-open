"""
Hierarchical Parameter Validator - Agent 01 Phase 2
===================================================

Validates extracted hierarchical parameters for correctness,
consistency, and musical validity.

Author: Agent 01 - Parameter Consolidation Architect
Date: November 20, 2025
Version: 2.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    score: float = 1.0  # 0.0-1.0, lower if warnings/errors


class HierarchicalParameterValidator:
    """
    Validates hierarchical parameter extractions.

    Checks:
    1. Type validity (correct types, ranges)
    2. Cross-parameter consistency (dependencies, correlations)
    3. Musical validity (reasonable values for genre)
    4. Hierarchical coherence (Level 3 matches Level 1 genre)
    """

    def __init__(self):
        self._load_schema()
        self._load_validation_rules()

    def _load_schema(self):
        """Load hierarchical parameter schema"""
        schema_path = Path(__file__).parent / "hierarchical_parameters.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        else:
            self.schema = {}

    def _load_validation_rules(self):
        """Load validation rules from schema"""
        self.validation_rules = self.schema.get('validation_rules', {}).get('rules', [])

    def validate_all(self, params: Dict[str, Any]) -> ValidationResult:
        """
        Validate all parameters comprehensively.

        Args:
            params: Hierarchical parameter dictionary

        Returns:
            ValidationResult with errors, warnings, and score
        """
        result = ValidationResult(is_valid=True)

        # Level 1 validation
        level1_result = self.validate_level1(params.get('level1_global', {}))
        result.errors.extend(level1_result.errors)
        result.warnings.extend(level1_result.warnings)

        # Level 2 validation
        level2_result = self.validate_level2(
            params.get('level2_universal', {}),
            params.get('level1_global', {})
        )
        result.errors.extend(level2_result.errors)
        result.warnings.extend(level2_result.warnings)

        # Level 3 validation
        level3_result = self.validate_level3(
            params.get('level3_genre_specific', {}),
            params.get('level1_global', {})
        )
        result.errors.extend(level3_result.errors)
        result.warnings.extend(level3_result.warnings)

        # Cross-level validation
        cross_result = self.validate_cross_parameter(params)
        result.errors.extend(cross_result.errors)
        result.warnings.extend(cross_result.warnings)

        # Determine overall validity
        if result.errors:
            result.is_valid = False

        # Compute score
        error_penalty = len(result.errors) * 0.2
        warning_penalty = len(result.warnings) * 0.05
        result.score = max(0.0, 1.0 - error_penalty - warning_penalty)

        return result

    def validate_level1(self, level1: Dict[str, Any]) -> ValidationResult:
        """Validate Level 1 global context parameters"""
        result = ValidationResult(is_valid=True)

        # tempo.bpm
        if 'tempo.bpm' in level1:
            tempo = level1['tempo.bpm']
            if not isinstance(tempo, (int, float)):
                result.errors.append(f"tempo.bpm must be numeric, got {type(tempo)}")
            elif not 40 <= tempo <= 200:
                result.warnings.append(f"tempo.bpm={tempo} outside typical range [40, 200]")

        # time_signature
        if 'time_signature' in level1:
            ts = level1['time_signature']
            valid_ts = ['4/4', '3/4', '6/8', '5/4', '7/8', '12/8', '2/4', '3/8', '9/8']
            if ts not in valid_ts:
                result.warnings.append(f"time_signature='{ts}' is unusual")

        # key.tonic
        if 'key.tonic' in level1:
            tonic = level1['key.tonic']
            valid_tonics = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            if tonic not in valid_tonics:
                result.errors.append(f"key.tonic='{tonic}' invalid")

        # key.mode
        if 'key.mode' in level1:
            mode = level1['key.mode']
            valid_modes = ['major', 'minor', 'dorian', 'phrygian', 'lydian', 'mixolydian', 'aeolian', 'locrian']
            if mode not in valid_modes:
                result.errors.append(f"key.mode='{mode}' invalid")

        # genre.primary
        if 'genre.primary' in level1:
            genre = level1['genre.primary']
            valid_genres = ['jazz', 'classical', 'rock', 'electronic', 'pop', 'latin', 'hiphop']
            if genre not in valid_genres:
                result.errors.append(f"genre.primary='{genre}' invalid")

        # energy.level
        if 'energy.level' in level1:
            energy = level1['energy.level']
            if not isinstance(energy, (int, float)):
                result.errors.append(f"energy.level must be numeric, got {type(energy)}")
            elif not 0.0 <= energy <= 1.0:
                result.errors.append(f"energy.level={energy} must be in [0.0, 1.0]")

        # complexity.overall
        if 'complexity.overall' in level1:
            complexity = level1['complexity.overall']
            if not isinstance(complexity, (int, float)):
                result.errors.append(f"complexity.overall must be numeric, got {type(complexity)}")
            elif not 0.0 <= complexity <= 1.0:
                result.errors.append(f"complexity.overall={complexity} must be in [0.0, 1.0]")

        # structure.form
        if 'structure.form' in level1:
            form = level1['structure.form']
            valid_forms = ['AABA', 'ABAC', 'verse_chorus', 'verse_chorus_bridge',
                          'through_composed', 'theme_variations', 'sonata', 'rondo']
            if form not in valid_forms:
                result.warnings.append(f"structure.form='{form}' is unusual")

        return result

    def validate_level2(self, level2: Dict[str, Any], level1: Dict[str, Any]) -> ValidationResult:
        """Validate Level 2 universal dimension parameters"""
        result = ValidationResult(is_valid=True)

        # Validate harmony parameters
        if 'harmony' in level2:
            result.errors.extend(self._validate_harmony(level2['harmony']))

        # Validate melody parameters
        if 'melody' in level2:
            result.errors.extend(self._validate_melody(level2['melody']))

        # Validate rhythm parameters
        if 'rhythm' in level2:
            result.errors.extend(self._validate_rhythm(level2['rhythm']))

        # Validate dynamics parameters
        if 'dynamics' in level2:
            result.errors.extend(self._validate_dynamics(level2['dynamics']))

        # Validate texture parameters
        if 'texture' in level2:
            result.errors.extend(self._validate_texture(level2['texture']))

        return result

    def _validate_harmony(self, harmony: Dict[str, Any]) -> List[str]:
        """Validate harmony parameters"""
        errors = []

        # chord_density
        if 'chord_density' in harmony:
            val = harmony['chord_density']
            if not isinstance(val, (int, float)) or val < 0 or val > 12:
                errors.append(f"harmony.chord_density={val} invalid (expect 0-12)")

        # complexity
        if 'complexity' in harmony:
            val = harmony['complexity']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"harmony.complexity={val} invalid (expect 0-1)")

        # chromaticism
        if 'chromaticism' in harmony:
            val = harmony['chromaticism']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"harmony.chromaticism={val} invalid (expect 0-1)")

        # tension
        if 'tension' in harmony:
            val = harmony['tension']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"harmony.tension={val} invalid (expect 0-1)")

        # voicing_spread
        if 'voicing_spread' in harmony:
            val = harmony['voicing_spread']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"harmony.voicing_spread={val} invalid (expect 0-1)")

        # progression_predictability
        if 'progression_predictability' in harmony:
            val = harmony['progression_predictability']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"harmony.progression_predictability={val} invalid (expect 0-1)")

        return errors

    def _validate_melody(self, melody: Dict[str, Any]) -> List[str]:
        """Validate melody parameters"""
        errors = []

        # note_density
        if 'note_density' in melody:
            val = melody['note_density']
            if not isinstance(val, (int, float)) or val < 0 or val > 16:
                errors.append(f"melody.note_density={val} invalid (expect 0-16)")

        # range_semitones
        if 'range_semitones' in melody:
            val = melody['range_semitones']
            if not isinstance(val, int) or val < 0 or val > 36:
                errors.append(f"melody.range_semitones={val} invalid (expect 0-36)")

        # contour_smoothness
        if 'contour_smoothness' in melody:
            val = melody['contour_smoothness']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"melody.contour_smoothness={val} invalid (expect 0-1)")

        # rhythmic_complexity
        if 'rhythmic_complexity' in melody:
            val = melody['rhythmic_complexity']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"melody.rhythmic_complexity={val} invalid (expect 0-1)")

        # repetition
        if 'repetition' in melody:
            val = melody['repetition']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"melody.repetition={val} invalid (expect 0-1)")

        return errors

    def _validate_rhythm(self, rhythm: Dict[str, Any]) -> List[str]:
        """Validate rhythm parameters"""
        errors = []

        # subdivision
        if 'subdivision' in rhythm:
            val = rhythm['subdivision']
            valid = ['whole', 'half', 'quarter', 'eighth', 'triplet', 'sixteenth', 'quintuplet', 'sextuplet']
            if val not in valid:
                errors.append(f"rhythm.subdivision='{val}' invalid")

        # syncopation
        if 'syncopation' in rhythm:
            val = rhythm['syncopation']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rhythm.syncopation={val} invalid (expect 0-1)")

        # groove_consistency
        if 'groove_consistency' in rhythm:
            val = rhythm['groove_consistency']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rhythm.groove_consistency={val} invalid (expect 0-1)")

        # polyrhythm
        if 'polyrhythm' in rhythm:
            val = rhythm['polyrhythm']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rhythm.polyrhythm={val} invalid (expect 0-1)")

        # swing_amount
        if 'swing_amount' in rhythm:
            val = rhythm['swing_amount']
            if not isinstance(val, (int, float)) or val < 0.5 or val > 0.75:
                errors.append(f"rhythm.swing_amount={val} invalid (expect 0.5-0.75)")

        return errors

    def _validate_dynamics(self, dynamics: Dict[str, Any]) -> List[str]:
        """Validate dynamics parameters"""
        errors = []

        # overall_level
        if 'overall_level' in dynamics:
            val = dynamics['overall_level']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"dynamics.overall_level={val} invalid (expect 0-1)")

        # range
        if 'range' in dynamics:
            val = dynamics['range']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"dynamics.range={val} invalid (expect 0-1)")

        return errors

    def _validate_texture(self, texture: Dict[str, Any]) -> List[str]:
        """Validate texture parameters"""
        errors = []

        # polyphony
        if 'polyphony' in texture:
            val = texture['polyphony']
            if not isinstance(val, int) or val < 1 or val > 12:
                errors.append(f"texture.polyphony={val} invalid (expect 1-12)")

        # density
        if 'density' in texture:
            val = texture['density']
            if not isinstance(val, (int, float)) or val < 0 or val > 20:
                errors.append(f"texture.density={val} invalid (expect 0-20)")

        return errors

    def validate_level3(self, level3: Dict[str, Any], level1: Dict[str, Any]) -> ValidationResult:
        """Validate Level 3 genre-specific parameters"""
        result = ValidationResult(is_valid=True)

        genre = level1.get('genre.primary', 'pop')

        # Check that genre-specific params match genre
        if 'jazz' in level3 and genre != 'jazz':
            result.warnings.append(f"jazz parameters present but genre='{genre}'")

        if 'classical' in level3 and genre != 'classical':
            result.warnings.append(f"classical parameters present but genre='{genre}'")

        if 'rock' in level3 and genre != 'rock':
            result.warnings.append(f"rock parameters present but genre='{genre}'")

        if 'electronic' in level3 and genre != 'electronic':
            result.warnings.append(f"electronic parameters present but genre='{genre}'")

        if 'hiphop' in level3 and genre != 'hiphop':
            result.warnings.append(f"hiphop parameters present but genre='{genre}'")

        if 'latin' in level3 and genre != 'latin':
            result.warnings.append(f"latin parameters present but genre='{genre}'")

        # Validate orchestration (always present)
        if 'orchestration' in level3:
            orch = level3['orchestration']

            if 'instrument_count' in orch:
                val = orch['instrument_count']
                if not isinstance(val, int) or val < 1 or val > 20:
                    result.errors.append(f"orchestration.instrument_count={val} invalid (expect 1-20)")

            if 'register_balance' in orch:
                val = orch['register_balance']
                if not isinstance(val, (int, float)) or val < 0 or val > 1:
                    result.errors.append(f"orchestration.register_balance={val} invalid (expect 0-1)")

            if 'legato_ratio' in orch:
                val = orch['legato_ratio']
                if not isinstance(val, (int, float)) or val < 0.3 or val > 1.0:
                    result.errors.append(f"orchestration.legato_ratio={val} invalid (expect 0.3-1.0)")

        # Validate genre-specific parameters
        if 'jazz' in level3:
            result.errors.extend(self._validate_jazz(level3['jazz']))

        if 'classical' in level3:
            result.errors.extend(self._validate_classical(level3['classical']))

        if 'rock' in level3:
            result.errors.extend(self._validate_rock(level3['rock']))

        return result

    def _validate_jazz(self, jazz: Dict[str, Any]) -> List[str]:
        """Validate jazz-specific parameters"""
        errors = []

        if 'swing_feel' in jazz:
            val = jazz['swing_feel']
            if val not in ['straight', 'light', 'medium', 'hard']:
                errors.append(f"jazz.swing_feel='{val}' invalid")

        if 'walking_bass' in jazz:
            val = jazz['walking_bass']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"jazz.walking_bass={val} invalid (expect 0-1)")

        if 'improvisation_ratio' in jazz:
            val = jazz['improvisation_ratio']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"jazz.improvisation_ratio={val} invalid (expect 0-1)")

        if 'bebop_vocabulary' in jazz:
            val = jazz['bebop_vocabulary']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"jazz.bebop_vocabulary={val} invalid (expect 0-1)")

        return errors

    def _validate_classical(self, classical: Dict[str, Any]) -> List[str]:
        """Validate classical-specific parameters"""
        errors = []

        if 'counterpoint' in classical:
            val = classical['counterpoint']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"classical.counterpoint={val} invalid (expect 0-1)")

        if 'development_density' in classical:
            val = classical['development_density']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"classical.development_density={val} invalid (expect 0-1)")

        if 'voice_leading_quality' in classical:
            val = classical['voice_leading_quality']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"classical.voice_leading_quality={val} invalid (expect 0-1)")

        return errors

    def _validate_rock(self, rock: Dict[str, Any]) -> List[str]:
        """Validate rock-specific parameters"""
        errors = []

        if 'power_chord_ratio' in rock:
            val = rock['power_chord_ratio']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rock.power_chord_ratio={val} invalid (expect 0-1)")

        if 'riff_repetition' in rock:
            val = rock['riff_repetition']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rock.riff_repetition={val} invalid (expect 0-1)")

        if 'distortion_level' in rock:
            val = rock['distortion_level']
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"rock.distortion_level={val} invalid (expect 0-1)")

        return errors

    def validate_cross_parameter(self, params: Dict[str, Any]) -> ValidationResult:
        """Validate cross-parameter consistency"""
        result = ValidationResult(is_valid=True)

        level1 = params.get('level1_global', {})
        level2 = params.get('level2_universal', {})
        level3 = params.get('level3_genre_specific', {})

        # Energy should correlate with dynamics
        if 'energy.level' in level1 and 'dynamics' in level2:
            energy = level1['energy.level']
            dynamics_level = level2['dynamics'].get('overall_level', 0.5)

            # They should be reasonably correlated
            if abs(energy - dynamics_level) > 0.5:
                result.warnings.append(
                    f"energy.level={energy:.2f} and dynamics.overall_level={dynamics_level:.2f} "
                    f"are not well correlated"
                )

        # Complexity should correlate with harmony complexity
        if 'complexity.overall' in level1 and 'harmony' in level2:
            complexity = level1['complexity.overall']
            harmony_complexity = level2['harmony'].get('complexity', 0.5)

            if abs(complexity - harmony_complexity) > 0.4:
                result.warnings.append(
                    f"complexity.overall={complexity:.2f} and harmony.complexity={harmony_complexity:.2f} "
                    f"should be more correlated"
                )

        # High swing amount should suggest jazz
        if 'rhythm' in level2 and level2['rhythm'].get('swing_amount', 0.5) > 0.65:
            if level1.get('genre.primary') not in ['jazz', 'blues']:
                result.warnings.append(
                    f"High swing_amount but genre is {level1.get('genre.primary')}"
                )

        # High power chord ratio should suggest rock
        if 'rock' in level3:
            power_chord_ratio = level3['rock'].get('power_chord_ratio', 0.0)
            if power_chord_ratio > 0.7:
                harmony_complexity = level2.get('harmony', {}).get('complexity', 0.5)
                if harmony_complexity > 0.6:
                    result.warnings.append(
                        f"High power_chord_ratio={power_chord_ratio:.2f} but "
                        f"harmony.complexity={harmony_complexity:.2f} is also high (unusual)"
                    )

        # High quantization should mean high groove consistency
        if 'electronic' in level3:
            quantization = level3['electronic'].get('quantization', 0.5)
            groove = level2.get('rhythm', {}).get('groove_consistency', 0.7)
            if quantization > 0.9 and groove < 0.7:
                result.warnings.append(
                    f"High quantization={quantization:.2f} but low groove_consistency={groove:.2f}"
                )

        return result


def test_validator():
    """Test the validator"""
    print("="*80)
    print("HIERARCHICAL PARAMETER VALIDATOR TEST")
    print("="*80)

    validator = HierarchicalParameterValidator()

    # Test valid parameters
    valid_params = {
        'level1_global': {
            'genre.primary': 'jazz',
            'tempo.bpm': 180.0,
            'time_signature': '4/4',
            'key.tonic': 'F',
            'key.mode': 'major',
            'energy.level': 0.7,
            'complexity.overall': 0.65,
            'structure.form': 'AABA'
        },
        'level2_universal': {
            'harmony': {
                'chord_density': 5.2,
                'complexity': 0.7,
                'chromaticism': 0.4,
                'tension': 0.5,
                'voicing_spread': 0.6,
                'progression_predictability': 0.6
            },
            'melody': {
                'note_density': 6.5,
                'range_semitones': 15,
                'contour_smoothness': 0.6,
                'rhythmic_complexity': 0.7,
                'repetition': 0.4
            },
            'rhythm': {
                'subdivision': 'sixteenth',
                'syncopation': 0.5,
                'groove_consistency': 0.8,
                'polyrhythm': 0.2,
                'swing_amount': 0.67
            },
            'dynamics': {
                'overall_level': 0.7,
                'range': 0.4
            },
            'texture': {
                'polyphony': 5,
                'density': 8.5
            }
        },
        'level3_genre_specific': {
            'orchestration': {
                'instrument_count': 5,
                'register_balance': 0.6,
                'legato_ratio': 0.85,
                'section_contrast': 0.5,
                'repetition_level': 0.5
            },
            'jazz': {
                'swing_feel': 'medium',
                'walking_bass': 0.9,
                'improvisation_ratio': 0.6,
                'bebop_vocabulary': 0.7
            }
        }
    }

    print("\n1. Validating VALID parameters:")
    result = validator.validate_all(valid_params)
    print(f"   Valid: {result.is_valid}")
    print(f"   Score: {result.score:.2f}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")

    if result.errors:
        print("\n   Errors:")
        for error in result.errors:
            print(f"     - {error}")

    if result.warnings:
        print("\n   Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")

    # Test invalid parameters
    print("\n2. Validating INVALID parameters:")
    invalid_params = {
        'level1_global': {
            'genre.primary': 'invalid_genre',  # Invalid
            'tempo.bpm': 500.0,  # Out of range (warning)
            'energy.level': 1.5,  # Out of range (error)
        },
        'level2_universal': {
            'harmony': {
                'complexity': 2.0,  # Out of range
            },
            'rhythm': {
                'subdivision': 'invalid',  # Invalid value
            }
        },
        'level3_genre_specific': {}
    }

    result = validator.validate_all(invalid_params)
    print(f"   Valid: {result.is_valid}")
    print(f"   Score: {result.score:.2f}")
    print(f"   Errors: {len(result.errors)}")
    print(f"   Warnings: {len(result.warnings)}")

    if result.errors:
        print("\n   Errors:")
        for error in result.errors:
            print(f"     - {error}")

    if result.warnings:
        print("\n   Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_validator()
