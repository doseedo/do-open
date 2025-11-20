"""
Legacy Parameter Adapter - Agent 01 Phase 2
===========================================

Provides backward compatibility between old 118-parameter system
and new 50-parameter hierarchical system.

This adapter enables:
1. old_to_new: Convert 118 legacy parameters → 50 hierarchical parameters
2. new_to_old: Convert 50 hierarchical parameters → 118 legacy parameters (lossy)

Author: Agent 01 - Parameter Consolidation Architect
Date: November 20, 2025
Version: 2.0.0
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np


class LegacyParameterAdapter:
    """
    Adapts between old (118-param) and new (50-param) systems.

    Usage:
        adapter = LegacyParameterAdapter()

        # Convert old → new
        new_params = adapter.old_to_new(old_params)

        # Convert new → old (lossy)
        old_params = adapter.new_to_old(new_params)
    """

    def __init__(self, show_deprecation_warnings: bool = True):
        self.show_warnings = show_deprecation_warnings
        self._load_migration_map()

    def _load_migration_map(self):
        """Load parameter migration map"""
        map_path = Path(__file__).parent / "parameter_migration_map.json"
        if map_path.exists():
            with open(map_path, 'r') as f:
                self.migration_map = json.load(f)
        else:
            warnings.warn("parameter_migration_map.json not found")
            self.migration_map = {}

    def old_to_new(self, old_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert old 118-parameter dict to new 50-parameter hierarchical dict.

        Args:
            old_params: Dictionary with old parameter names/values

        Returns:
            Dictionary with hierarchical structure (level1, level2, level3)
        """
        if self.show_warnings:
            warnings.warn(
                "Using legacy parameter API. This will be deprecated in 6 months. "
                "Please migrate to hierarchical parameter system.",
                DeprecationWarning,
                stacklevel=2
            )

        new_params = {
            'level1_global': {},
            'level2_universal': {
                'harmony': {},
                'melody': {},
                'rhythm': {},
                'dynamics': {},
                'texture': {}
            },
            'level3_genre_specific': {
                'orchestration': {}
            }
        }

        # ====================================================================
        # DIRECT MAPPINGS
        # ====================================================================

        # Rhythm - direct mappings
        if 'rhythm.swing.amount' in old_params:
            new_params['level2_universal']['rhythm']['swing_amount'] = old_params['rhythm.swing.amount']

        if 'rhythm.syncopation.probability' in old_params:
            new_params['level2_universal']['rhythm']['syncopation'] = old_params['rhythm.syncopation.probability']

        # Harmony - direct mappings
        if 'harmony.voicing.spread' in old_params:
            new_params['level2_universal']['harmony']['voicing_spread'] = old_params['harmony.voicing.spread']

        # Articulation - direct mapping
        if 'articulation.duration.ratio' in old_params:
            new_params['level3_genre_specific']['orchestration']['legato_ratio'] = old_params['articulation.duration.ratio']

        # Genre-specific - direct mappings
        if 'bass.style.walking_probability' in old_params:
            if 'jazz' not in new_params['level3_genre_specific']:
                new_params['level3_genre_specific']['jazz'] = {}
            new_params['level3_genre_specific']['jazz']['walking_bass'] = old_params['bass.style.walking_probability']

        if 'genre.rock.power_chord_probability' in old_params:
            if 'rock' not in new_params['level3_genre_specific']:
                new_params['level3_genre_specific']['rock'] = {}
            new_params['level3_genre_specific']['rock']['power_chord_ratio'] = old_params['genre.rock.power_chord_probability']

        # ====================================================================
        # MERGED MAPPINGS (N → 1)
        # ====================================================================

        # Harmony extensions → harmony.complexity
        use_9ths = float(old_params.get('harmony.extensions.use_9ths', True))
        use_11ths = float(old_params.get('harmony.extensions.use_11ths', True))
        use_13ths = float(old_params.get('harmony.extensions.use_13ths', False))
        new_params['level2_universal']['harmony']['complexity'] = (
            0.3 * use_9ths + 0.3 * use_11ths + 0.4 * use_13ths
        )

        # Harmony substitutions → harmony.chromaticism
        tritone = old_params.get('harmony.substitution.tritone_probability', 0.3)
        modal = old_params.get('harmony.substitution.modal_interchange_probability', 0.2)
        new_params['level2_universal']['harmony']['chromaticism'] = (tritone + modal) / 2

        # Melody intervals → melody.contour_smoothness
        stepwise = old_params.get('melody.intervals.stepwise_probability', 0.7)
        max_leap = old_params.get('melody.intervals.max_leap', 12)
        new_params['level2_universal']['melody']['contour_smoothness'] = (
            stepwise * (1 - max_leap / 24.0)
        )

        # Melody intervals → melody.range_semitones
        new_params['level2_universal']['melody']['range_semitones'] = max_leap

        # Dynamics → dynamics.overall_level
        velocity_base = old_params.get('dynamics.velocity.base', 80)
        new_params['level2_universal']['dynamics']['overall_level'] = velocity_base / 127.0

        # Dynamics → dynamics.range
        velocity_var = old_params.get('dynamics.velocity.variation', 20)
        new_params['level2_universal']['dynamics']['range'] = velocity_var / 127.0

        # Texture (consolidate multiple texture params)
        polyphonic_density = old_params.get('texture.polyphonic.density', 0.3)
        layering_count = old_params.get('texture.layering.count', 3)
        new_params['level2_universal']['texture']['polyphony'] = int(layering_count * 3)

        horizontal_density = old_params.get('texture.horizontal.density', 0.6)
        new_params['level2_universal']['texture']['density'] = horizontal_density * 10.0

        # Voicing → chord_density (heuristic)
        voicing_density = old_params.get('harmony.voicing.density', 4)
        new_params['level2_universal']['harmony']['chord_density'] = float(voicing_density)

        # ====================================================================
        # COMPUTED/NEW PARAMETERS (Defaults)
        # ====================================================================

        # Level 1 - use defaults, would need actual MIDI for extraction
        new_params['level1_global'] = {
            'genre.primary': 'jazz',  # Default
            'tempo.bpm': 120.0,
            'time_signature': '4/4',
            'key.tonic': 'C',
            'key.mode': 'major',
            'energy.level': 0.5,  # Would compute from dynamics + tempo + density
            'complexity.overall': 0.5,  # Would compute from harmony + melody
            'structure.form': 'AABA'
        }

        # Compute energy.level from available params
        if 'dynamics.velocity.base' in old_params and 'tempo.bpm' in old_params:
            dynamics_norm = old_params['dynamics.velocity.base'] / 127.0
            tempo_norm = min(old_params.get('tempo.bpm', 120) / 200.0, 1.0)
            texture_norm = min(horizontal_density, 1.0)
            new_params['level1_global']['energy.level'] = (
                0.3 * dynamics_norm + 0.3 * tempo_norm + 0.4 * texture_norm
            )

        # Compute complexity.overall
        harmony_complexity = new_params['level2_universal']['harmony']['complexity']
        new_params['level1_global']['complexity.overall'] = min(harmony_complexity * 1.2, 1.0)

        # Level 2 - fill in missing parameters with defaults
        level2_defaults = {
            'harmony': {
                'chord_density': 4.0,
                'complexity': 0.5,
                'chromaticism': 0.3,
                'tension': 0.5,
                'voicing_spread': 0.5,
                'progression_predictability': 0.5
            },
            'melody': {
                'note_density': 4.0,
                'range_semitones': 12,
                'contour_smoothness': 0.7,
                'rhythmic_complexity': 0.5,
                'repetition': 0.5
            },
            'rhythm': {
                'subdivision': 'eighth',
                'syncopation': 0.3,
                'groove_consistency': 0.7,
                'polyrhythm': 0.1,
                'swing_amount': 0.67
            },
            'dynamics': {
                'overall_level': 0.6,
                'range': 0.3
            },
            'texture': {
                'polyphony': 4,
                'density': 5.0
            }
        }

        # Fill in defaults for missing parameters
        for category, defaults in level2_defaults.items():
            for param, default_value in defaults.items():
                if param not in new_params['level2_universal'][category]:
                    new_params['level2_universal'][category][param] = default_value

        # Level 3 - universal orchestration
        orchestration_defaults = {
            'instrument_count': 5,
            'register_balance': 0.5,
            'legato_ratio': 0.9,
            'section_contrast': 0.5,
            'repetition_level': 0.5
        }

        for param, default_value in orchestration_defaults.items():
            if param not in new_params['level3_genre_specific']['orchestration']:
                new_params['level3_genre_specific']['orchestration'][param] = default_value

        return new_params

    def new_to_old(self, new_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert new 50-parameter hierarchical dict to old 118-parameter dict.

        WARNING: This is a lossy conversion. Dropped parameters are set to defaults.

        Args:
            new_params: Dictionary with hierarchical structure

        Returns:
            Dictionary with old parameter names/values
        """
        old_params = {}

        level1 = new_params.get('level1_global', {})
        level2 = new_params.get('level2_universal', {})
        level3 = new_params.get('level3_genre_specific', {})

        # ====================================================================
        # REVERSE DIRECT MAPPINGS
        # ====================================================================

        # Rhythm
        if 'rhythm' in level2:
            if 'swing_amount' in level2['rhythm']:
                old_params['rhythm.swing.amount'] = level2['rhythm']['swing_amount']
            if 'syncopation' in level2['rhythm']:
                old_params['rhythm.syncopation.probability'] = level2['rhythm']['syncopation']

        # Harmony
        if 'harmony' in level2:
            if 'voicing_spread' in level2['harmony']:
                old_params['harmony.voicing.spread'] = level2['harmony']['voicing_spread']

        # Articulation
        if 'orchestration' in level3 and 'legato_ratio' in level3['orchestration']:
            old_params['articulation.duration.ratio'] = level3['orchestration']['legato_ratio']

        # Genre-specific
        if 'jazz' in level3:
            if 'walking_bass' in level3['jazz']:
                old_params['bass.style.walking_probability'] = level3['jazz']['walking_bass']

        if 'rock' in level3:
            if 'power_chord_ratio' in level3['rock']:
                old_params['genre.rock.power_chord_probability'] = level3['rock']['power_chord_ratio']

        # ====================================================================
        # REVERSE MERGED MAPPINGS (1 → N)
        # ====================================================================

        # Harmony complexity → extensions (reverse formula)
        if 'harmony' in level2 and 'complexity' in level2['harmony']:
            complexity = level2['harmony']['complexity']
            # Heuristic reverse mapping
            old_params['harmony.extensions.use_9ths'] = complexity > 0.3
            old_params['harmony.extensions.use_11ths'] = complexity > 0.5
            old_params['harmony.extensions.use_13ths'] = complexity > 0.7

        # Harmony chromaticism → substitutions
        if 'harmony' in level2 and 'chromaticism' in level2['harmony']:
            chromaticism = level2['harmony']['chromaticism']
            old_params['harmony.substitution.tritone_probability'] = chromaticism
            old_params['harmony.substitution.modal_interchange_probability'] = chromaticism

        # Melody contour_smoothness → intervals (reverse)
        if 'melody' in level2:
            if 'contour_smoothness' in level2['melody']:
                smoothness = level2['melody']['contour_smoothness']
                old_params['melody.intervals.stepwise_probability'] = min(smoothness * 1.2, 1.0)

            if 'range_semitones' in level2['melody']:
                old_params['melody.intervals.max_leap'] = level2['melody']['range_semitones']

        # Dynamics
        if 'dynamics' in level2:
            if 'overall_level' in level2['dynamics']:
                old_params['dynamics.velocity.base'] = int(level2['dynamics']['overall_level'] * 127)
            if 'range' in level2['dynamics']:
                old_params['dynamics.velocity.variation'] = int(level2['dynamics']['range'] * 127)

        # Texture → multiple old params
        if 'texture' in level2:
            if 'polyphony' in level2['texture']:
                polyphony = level2['texture']['polyphony']
                old_params['texture.polyphonic.density'] = min(polyphony / 10.0, 1.0)
                old_params['texture.layering.count'] = max(polyphony // 3, 1)

            if 'density' in level2['texture']:
                old_params['texture.horizontal.density'] = min(level2['texture']['density'] / 10.0, 1.0)

        # Voicing
        if 'harmony' in level2 and 'chord_density' in level2['harmony']:
            old_params['harmony.voicing.density'] = int(min(level2['harmony']['chord_density'], 7))

        # ====================================================================
        # DROPPED PARAMETERS (Set to defaults)
        # ====================================================================

        # Voice leading
        old_params['harmony.voice_leading.smoothness'] = 0.8
        old_params['harmony.voice_leading.parallel_motion_tolerance'] = 0.1

        # Melody
        old_params['melody.contour.type'] = 'arch'
        old_params['melody.chromaticism.amount'] = level2.get('harmony', {}).get('chromaticism', 0.3)
        old_params['melody.ornaments.probability'] = 0.2

        # Rhythm
        old_params['rhythm.microtiming.variation'] = 10

        # Drums (defaults)
        old_params['drums.kick.velocity_min'] = 80
        old_params['drums.kick.velocity_max'] = 110

        # Genre
        old_params['genre.rock.bend_probability'] = 0.3
        old_params['genre.rock.vibrato_probability'] = 0.4
        old_params['genre.rock.vibrato_depth'] = 30.0

        # Texture (additional defaults)
        old_params['texture.voice.independence'] = 0.5
        old_params['texture.vertical.density'] = 0.4
        old_params['texture.register.spread'] = 0.5
        old_params['texture.contrast.rate'] = 0.3
        old_params['texture.homophonic.ratio'] = 0.4
        old_params['texture.voice_crossing.density'] = 0.1
        old_params['texture.rhythmic.independence'] = 0.5

        # Voicing (additional)
        old_params['harmony.voicing.type'] = 'close'

        return old_params

    def validate_conversion(self, old_params: Dict[str, Any],
                          new_params: Dict[str, Any],
                          tolerance: float = 0.1) -> Dict[str, Any]:
        """
        Validate that old→new→old conversion preserves important values.

        Args:
            old_params: Original old parameters
            new_params: Converted new parameters
            tolerance: Acceptable difference for continuous values

        Returns:
            Dictionary with validation results
        """
        # Convert back
        reconstructed_old = self.new_to_old(new_params)

        results = {
            'total_checked': 0,
            'preserved': 0,
            'changed': 0,
            'errors': []
        }

        # Check important parameters
        important_params = [
            'rhythm.swing.amount',
            'rhythm.syncopation.probability',
            'harmony.voicing.spread',
            'dynamics.velocity.base',
            'articulation.duration.ratio'
        ]

        for param in important_params:
            if param in old_params and param in reconstructed_old:
                results['total_checked'] += 1
                old_val = old_params[param]
                new_val = reconstructed_old[param]

                if isinstance(old_val, (int, float)):
                    diff = abs(old_val - new_val)
                    if diff <= tolerance * abs(old_val) if old_val != 0 else tolerance:
                        results['preserved'] += 1
                    else:
                        results['changed'] += 1
                        results['errors'].append({
                            'parameter': param,
                            'old_value': old_val,
                            'new_value': new_val,
                            'difference': diff
                        })
                else:
                    if old_val == new_val:
                        results['preserved'] += 1
                    else:
                        results['changed'] += 1
                        results['errors'].append({
                            'parameter': param,
                            'old_value': old_val,
                            'new_value': new_val
                        })

        results['preservation_rate'] = (
            results['preserved'] / results['total_checked']
            if results['total_checked'] > 0 else 0.0
        )

        return results


def test_adapter():
    """Test the adapter with sample data"""
    print("="*80)
    print("LEGACY PARAMETER ADAPTER TEST")
    print("="*80)

    adapter = LegacyParameterAdapter(show_deprecation_warnings=False)

    # Sample old parameters
    old_params = {
        'rhythm.swing.amount': 0.67,
        'rhythm.syncopation.probability': 0.4,
        'harmony.voicing.spread': 0.6,
        'harmony.extensions.use_9ths': True,
        'harmony.extensions.use_11ths': True,
        'harmony.extensions.use_13ths': False,
        'harmony.substitution.tritone_probability': 0.3,
        'harmony.substitution.modal_interchange_probability': 0.2,
        'melody.intervals.stepwise_probability': 0.8,
        'melody.intervals.max_leap': 10,
        'dynamics.velocity.base': 90,
        'dynamics.velocity.variation': 25,
        'bass.style.walking_probability': 0.85,
        'genre.rock.power_chord_probability': 0.7,
        'articulation.duration.ratio': 0.95
    }

    print("\n1. Converting OLD → NEW:")
    new_params = adapter.old_to_new(old_params)

    print("\nLevel 1 (Global Context):")
    for key, value in new_params['level1_global'].items():
        print(f"  {key:30s} = {value}")

    print("\nLevel 2 (Universal Dimensions):")
    for category, subparams in new_params['level2_universal'].items():
        print(f"  {category}:")
        for key, value in subparams.items():
            print(f"    {key:28s} = {value}")

    print("\n2. Converting NEW → OLD:")
    reconstructed = adapter.new_to_old(new_params)

    print("\nReconstructed old parameters (sample):")
    for key in list(old_params.keys())[:10]:
        if key in reconstructed:
            print(f"  {key:50s} = {reconstructed[key]}")

    print("\n3. Validation:")
    validation = adapter.validate_conversion(old_params, new_params)
    print(f"  Total checked: {validation['total_checked']}")
    print(f"  Preserved: {validation['preserved']}")
    print(f"  Changed: {validation['changed']}")
    print(f"  Preservation rate: {validation['preservation_rate']:.1%}")

    if validation['errors']:
        print("\n  Errors:")
        for error in validation['errors'][:5]:  # Show first 5
            print(f"    {error['parameter']}: {error['old_value']} → {error['new_value']}")

    print("\n" + "="*80)


if __name__ == "__main__":
    test_adapter()
