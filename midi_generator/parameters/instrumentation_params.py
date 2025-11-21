#!/usr/bin/env python3
"""
Instrumentation & Orchestration Parameters - Agent 7
=====================================================

Defines ~50 parameters for instrumentation and orchestration decisions.
These parameters control how music is distributed across instruments,
how voicings are created, and how orchestral balance is achieved.

Target: 50 parameters
Files refactored: orchestrator.py, instrument_library.py

Author: Agent 7 - Instrumentation & Orchestration
Part of: Focused Parameter Refactoring (Agents 1-10)
"""

from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
from enum import Enum


# =============================================================================
# PARAMETER TYPES
# =============================================================================

class ParameterType(Enum):
    """Parameter value types"""
    CONTINUOUS = "continuous"      # Float in range
    DISCRETE = "discrete"          # Integer in range
    CATEGORICAL = "categorical"    # One of several options
    BOOLEAN = "boolean"            # True/False


@dataclass
class Parameter:
    """Represents a single parameter"""
    name: str
    type: ParameterType
    default: Any
    range: Tuple[Any, Any] = None  # For continuous/discrete
    options: List[Any] = None       # For categorical
    description: str = ""
    musical_impact: str = "medium"  # low, medium, high
    genre_relevance: List[str] = None  # Which genres use this heavily


# =============================================================================
# INSTRUMENTATION PARAMETERS (50 total)
# =============================================================================

INSTRUMENTATION_PARAMETERS = {

    # -----------------------------------------------------
    # DOUBLING PARAMETERS (10 params)
    # -----------------------------------------------------

    "instrumentation.doubling.octave_probability": Parameter(
        name="instrumentation.doubling.octave_probability",
        type=ParameterType.CONTINUOUS,
        default=0.4,
        range=(0.0, 1.0),
        description="Probability of doubling a voice at the octave",
        musical_impact="high",
        genre_relevance=["orchestral", "film", "romantic"]
    ),

    "instrumentation.doubling.unison_probability": Parameter(
        name="instrumentation.doubling.unison_probability",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Probability of doubling a voice in unison",
        musical_impact="medium",
        genre_relevance=["classical", "chamber"]
    ),

    "instrumentation.doubling.two_octave_probability": Parameter(
        name="instrumentation.doubling.two_octave_probability",
        type=ParameterType.CONTINUOUS,
        default=0.15,
        range=(0.0, 1.0),
        description="Probability of doubling two octaves apart",
        musical_impact="medium",
        genre_relevance=["romantic", "film"]
    ),

    "instrumentation.doubling.family_preference": Parameter(
        name="instrumentation.doubling.family_preference",
        type=ParameterType.CATEGORICAL,
        default="within",
        options=["within", "across", "mixed"],
        description="Prefer doubling within instrument family or across families",
        musical_impact="high",
        genre_relevance=["orchestral"]
    ),

    "instrumentation.doubling.bass_reinforcement": Parameter(
        name="instrumentation.doubling.bass_reinforcement",
        type=ParameterType.CONTINUOUS,
        default=0.7,
        range=(0.0, 1.0),
        description="Tendency to reinforce bass line with multiple instruments",
        musical_impact="high",
        genre_relevance=["orchestral", "band"]
    ),

    "instrumentation.doubling.melody_reinforcement": Parameter(
        name="instrumentation.doubling.melody_reinforcement",
        type=ParameterType.CONTINUOUS,
        default=0.6,
        range=(0.0, 1.0),
        description="Tendency to double melody line",
        musical_impact="high",
        genre_relevance=["orchestral", "film"]
    ),

    "instrumentation.doubling.thirds_probability": Parameter(
        name="instrumentation.doubling.thirds_probability",
        type=ParameterType.CONTINUOUS,
        default=0.25,
        range=(0.0, 1.0),
        description="Probability of doubling in parallel thirds (e.g., flute + clarinet)",
        musical_impact="medium",
        genre_relevance=["classical", "romantic"]
    ),

    "instrumentation.doubling.sixths_probability": Parameter(
        name="instrumentation.doubling.sixths_probability",
        type=ParameterType.CONTINUOUS,
        default=0.2,
        range=(0.0, 1.0),
        description="Probability of doubling in parallel sixths",
        musical_impact="medium",
        genre_relevance=["classical"]
    ),

    "instrumentation.doubling.avoid_muddy_bass": Parameter(
        name="instrumentation.doubling.avoid_muddy_bass",
        type=ParameterType.BOOLEAN,
        default=True,
        description="Avoid close doubling in low register (< C3)",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.doubling.max_simultaneous": Parameter(
        name="instrumentation.doubling.max_simultaneous",
        type=ParameterType.DISCRETE,
        default=3,
        range=(1, 6),
        description="Maximum instruments doubling same pitch/octave",
        musical_impact="medium",
        genre_relevance=["orchestral"]
    ),

    # -----------------------------------------------------
    # VOICING PARAMETERS (12 params)
    # -----------------------------------------------------

    "instrumentation.voicing.close_position_ratio": Parameter(
        name="instrumentation.voicing.close_position_ratio",
        type=ParameterType.CONTINUOUS,
        default=0.5,
        range=(0.0, 1.0),
        description="0.0=wide/open position, 1.0=close position",
        musical_impact="high",
        genre_relevance=["classical", "jazz", "chamber"]
    ),

    "instrumentation.voicing.max_spacing_upper": Parameter(
        name="instrumentation.voicing.max_spacing_upper",
        type=ParameterType.DISCRETE,
        default=12,
        range=(5, 24),
        description="Maximum semitones between upper voices",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.voicing.max_spacing_bass_tenor": Parameter(
        name="instrumentation.voicing.max_spacing_bass_tenor",
        type=ParameterType.DISCRETE,
        default=19,
        range=(12, 36),
        description="Maximum semitones between bass and tenor (can be wider)",
        musical_impact="medium",
        genre_relevance=["orchestral"]
    ),

    "instrumentation.voicing.drop_voicing_probability": Parameter(
        name="instrumentation.voicing.drop_voicing_probability",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Probability of using drop voicings (drop-2, drop-3)",
        musical_impact="medium",
        genre_relevance=["jazz", "contemporary"]
    ),

    "instrumentation.voicing.quartal_probability": Parameter(
        name="instrumentation.voicing.quartal_probability",
        type=ParameterType.CONTINUOUS,
        default=0.1,
        range=(0.0, 1.0),
        description="Probability of using quartal/quintal voicings",
        musical_impact="medium",
        genre_relevance=["jazz", "contemporary", "film"]
    ),

    "instrumentation.voicing.cluster_probability": Parameter(
        name="instrumentation.voicing.cluster_probability",
        type=ParameterType.CONTINUOUS,
        default=0.05,
        range=(0.0, 1.0),
        description="Probability of tone clusters (contemporary)",
        musical_impact="low",
        genre_relevance=["contemporary", "avant-garde"]
    ),

    "instrumentation.voicing.spread_factor": Parameter(
        name="instrumentation.voicing.spread_factor",
        type=ParameterType.CONTINUOUS,
        default=0.5,
        range=(0.0, 1.0),
        description="How spread out voicing is across registers (0=compact, 1=wide)",
        musical_impact="high",
        genre_relevance=["orchestral", "film"]
    ),

    "instrumentation.voicing.density": Parameter(
        name="instrumentation.voicing.density",
        type=ParameterType.DISCRETE,
        default=4,
        range=(2, 8),
        description="Number of simultaneous voices in voicing",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.voicing.root_position_preference": Parameter(
        name="instrumentation.voicing.root_position_preference",
        type=ParameterType.CONTINUOUS,
        default=0.4,
        range=(0.0, 1.0),
        description="Tendency to include root in bass (0=rootless, 1=always root)",
        musical_impact="medium",
        genre_relevance=["classical", "jazz"]
    ),

    "instrumentation.voicing.tessitura_balance": Parameter(
        name="instrumentation.voicing.tessitura_balance",
        type=ParameterType.CONTINUOUS,
        default=0.6,
        range=(0.0, 1.0),
        description="Balance between comfortable (1.0) and extreme (0.0) registers",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.voicing.voice_crossing_tolerance": Parameter(
        name="instrumentation.voicing.voice_crossing_tolerance",
        type=ParameterType.CONTINUOUS,
        default=0.2,
        range=(0.0, 1.0),
        description="Tolerance for voice crossing (0=strict, 1=free)",
        musical_impact="medium",
        genre_relevance=["contemporary", "jazz"]
    ),

    "instrumentation.voicing.omit_fifth_probability": Parameter(
        name="instrumentation.voicing.omit_fifth_probability",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Probability of omitting fifth from chord voicing",
        musical_impact="medium",
        genre_relevance=["jazz", "contemporary"]
    ),

    # -----------------------------------------------------
    # DYNAMICS & BALANCE PARAMETERS (8 params)
    # -----------------------------------------------------

    "instrumentation.dynamics.pp_to_ff_range": Parameter(
        name="instrumentation.dynamics.pp_to_ff_range",
        type=ParameterType.CONTINUOUS,
        default=0.7,
        range=(0.0, 1.0),
        description="Dynamic range usage (0=narrow, 1=full pp to ff)",
        musical_impact="high",
        genre_relevance=["orchestral", "chamber"]
    ),

    "instrumentation.dynamics.balance_melody_ratio": Parameter(
        name="instrumentation.dynamics.balance_melody_ratio",
        type=ParameterType.CONTINUOUS,
        default=1.3,
        range=(1.0, 2.0),
        description="Melody dynamic level relative to accompaniment",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.dynamics.balance_bass_ratio": Parameter(
        name="instrumentation.dynamics.balance_bass_ratio",
        type=ParameterType.CONTINUOUS,
        default=1.1,
        range=(0.8, 1.5),
        description="Bass dynamic level relative to middle voices",
        musical_impact="medium",
        genre_relevance=["orchestral", "band"]
    ),

    "instrumentation.dynamics.family_balance_mode": Parameter(
        name="instrumentation.dynamics.family_balance_mode",
        type=ParameterType.CATEGORICAL,
        default="auto",
        options=["auto", "equal", "weighted", "custom"],
        description="How to balance different instrument families",
        musical_impact="high",
        genre_relevance=["orchestral"]
    ),

    "instrumentation.dynamics.swell_probability": Parameter(
        name="instrumentation.dynamics.swell_probability",
        type=ParameterType.CONTINUOUS,
        default=0.2,
        range=(0.0, 1.0),
        description="Probability of crescendo-diminuendo swells",
        musical_impact="medium",
        genre_relevance=["romantic", "film"]
    ),

    "instrumentation.dynamics.accent_strength": Parameter(
        name="instrumentation.dynamics.accent_strength",
        type=ParameterType.CONTINUOUS,
        default=1.3,
        range=(1.0, 2.0),
        description="Multiplier for accented notes",
        musical_impact="medium",
        genre_relevance=["all"]
    ),

    "instrumentation.dynamics.layer_reduction_factor": Parameter(
        name="instrumentation.dynamics.layer_reduction_factor",
        type=ParameterType.CONTINUOUS,
        default=0.9,
        range=(0.7, 1.0),
        description="Reduce dynamics for each additional doubling layer",
        musical_impact="medium",
        genre_relevance=["orchestral"]
    ),

    "instrumentation.dynamics.register_compensation": Parameter(
        name="instrumentation.dynamics.register_compensation",
        type=ParameterType.BOOLEAN,
        default=True,
        description="Adjust dynamics based on register (louder in extremes)",
        musical_impact="medium",
        genre_relevance=["orchestral"]
    ),

    # -----------------------------------------------------
    # INSTRUMENT SELECTION PARAMETERS (10 params)
    # -----------------------------------------------------

    "instrumentation.selection.prefer_strings_probability": Parameter(
        name="instrumentation.selection.prefer_strings_probability",
        type=ParameterType.CONTINUOUS,
        default=0.5,
        range=(0.0, 1.0),
        description="Preference for using string instruments",
        musical_impact="high",
        genre_relevance=["orchestral", "chamber"]
    ),

    "instrumentation.selection.prefer_winds_probability": Parameter(
        name="instrumentation.selection.prefer_winds_probability",
        type=ParameterType.CONTINUOUS,
        default=0.4,
        range=(0.0, 1.0),
        description="Preference for using wind instruments",
        musical_impact="medium",
        genre_relevance=["orchestral", "band"]
    ),

    "instrumentation.selection.prefer_brass_probability": Parameter(
        name="instrumentation.selection.prefer_brass_probability",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Preference for using brass instruments",
        musical_impact="medium",
        genre_relevance=["orchestral", "band", "film"]
    ),

    "instrumentation.selection.homogeneous_blend_probability": Parameter(
        name="instrumentation.selection.homogeneous_blend_probability",
        type=ParameterType.CONTINUOUS,
        default=0.6,
        range=(0.0, 1.0),
        description="Prefer instruments from same family for blend",
        musical_impact="high",
        genre_relevance=["classical"]
    ),

    "instrumentation.selection.heterogeneous_color_probability": Parameter(
        name="instrumentation.selection.heterogeneous_color_probability",
        type=ParameterType.CONTINUOUS,
        default=0.4,
        range=(0.0, 1.0),
        description="Mix different families for color",
        musical_impact="medium",
        genre_relevance=["romantic", "film"]
    ),

    "instrumentation.selection.solo_vs_ensemble_ratio": Parameter(
        name="instrumentation.selection.solo_vs_ensemble_ratio",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Ratio of solo passages to ensemble (0=all ensemble, 1=all solo)",
        musical_impact="high",
        genre_relevance=["orchestral", "chamber"]
    ),

    "instrumentation.selection.rare_instrument_probability": Parameter(
        name="instrumentation.selection.rare_instrument_probability",
        type=ParameterType.CONTINUOUS,
        default=0.1,
        range=(0.0, 1.0),
        description="Probability of using uncommon instruments (harp, celesta, etc.)",
        musical_impact="low",
        genre_relevance=["film", "contemporary"]
    ),

    "instrumentation.selection.percussion_density": Parameter(
        name="instrumentation.selection.percussion_density",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Amount of percussion activity (0=sparse, 1=heavy)",
        musical_impact="medium",
        genre_relevance=["orchestral", "film", "contemporary"]
    ),

    "instrumentation.selection.orchestration_size": Parameter(
        name="instrumentation.selection.orchestration_size",
        type=ParameterType.CATEGORICAL,
        default="chamber",
        options=["solo", "chamber", "small_ensemble", "orchestra", "large_orchestra"],
        description="Size of ensemble to use",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.selection.period_style": Parameter(
        name="instrumentation.selection.period_style",
        type=ParameterType.CATEGORICAL,
        default="classical",
        options=["baroque", "classical", "romantic", "modern", "film", "contemporary"],
        description="Historical period for instrumentation choices",
        musical_impact="high",
        genre_relevance=["orchestral"]
    ),

    # -----------------------------------------------------
    # REGISTER & SPACING PARAMETERS (5 params)
    # -----------------------------------------------------

    "instrumentation.register.prefer_high_melody": Parameter(
        name="instrumentation.register.prefer_high_melody",
        type=ParameterType.BOOLEAN,
        default=True,
        description="Prefer melody in upper register",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.register.bass_octave_preference": Parameter(
        name="instrumentation.register.bass_octave_preference",
        type=ParameterType.DISCRETE,
        default=2,
        range=(1, 4),
        description="Preferred octave for bass (1=very low, 4=higher)",
        musical_impact="high",
        genre_relevance=["all"]
    ),

    "instrumentation.register.middle_voice_density": Parameter(
        name="instrumentation.register.middle_voice_density",
        type=ParameterType.CONTINUOUS,
        default=0.6,
        range=(0.0, 1.0),
        description="How filled-in middle voices are (0=sparse, 1=dense)",
        musical_impact="medium",
        genre_relevance=["orchestral", "romantic"]
    ),

    "instrumentation.register.extreme_register_probability": Parameter(
        name="instrumentation.register.extreme_register_probability",
        type=ParameterType.CONTINUOUS,
        default=0.15,
        range=(0.0, 1.0),
        description="Probability of using extreme high/low registers",
        musical_impact="medium",
        genre_relevance=["romantic", "film", "contemporary"]
    ),

    "instrumentation.register.octave_displacement_probability": Parameter(
        name="instrumentation.register.octave_displacement_probability",
        type=ParameterType.CONTINUOUS,
        default=0.2,
        range=(0.0, 1.0),
        description="Probability of displacing voice by octave for effect",
        musical_impact="medium",
        genre_relevance=["contemporary", "jazz"]
    ),

    # -----------------------------------------------------
    # ORCHESTRATION TECHNIQUE PARAMETERS (5 params)
    # -----------------------------------------------------

    "instrumentation.technique.arpeggiation_probability": Parameter(
        name="instrumentation.technique.arpeggiation_probability",
        type=ParameterType.CONTINUOUS,
        default=0.2,
        range=(0.0, 1.0),
        description="Probability of arpeggiating chords instead of block voicing",
        musical_impact="medium",
        genre_relevance=["classical", "romantic"]
    ),

    "instrumentation.technique.divisi_probability": Parameter(
        name="instrumentation.technique.divisi_probability",
        type=ParameterType.CONTINUOUS,
        default=0.3,
        range=(0.0, 1.0),
        description="Probability of string section divisi",
        musical_impact="medium",
        genre_relevance=["orchestral", "film"]
    ),

    "instrumentation.technique.tremolo_probability": Parameter(
        name="instrumentation.technique.tremolo_probability",
        type=ParameterType.CONTINUOUS,
        default=0.15,
        range=(0.0, 1.0),
        description="Probability of tremolo articulation",
        musical_impact="low",
        genre_relevance=["romantic", "film"]
    ),

    "instrumentation.technique.pizzicato_probability": Parameter(
        name="instrumentation.technique.pizzicato_probability",
        type=ParameterType.CONTINUOUS,
        default=0.1,
        range=(0.0, 1.0),
        description="Probability of pizzicato in string parts",
        musical_impact="low",
        genre_relevance=["orchestral", "chamber"]
    ),

    "instrumentation.technique.muting_probability": Parameter(
        name="instrumentation.technique.muting_probability",
        type=ParameterType.CONTINUOUS,
        default=0.1,
        range=(0.0, 1.0),
        description="Probability of using mutes (brass/strings)",
        musical_impact="low",
        genre_relevance=["jazz", "contemporary"]
    ),
}


# =============================================================================
# PARAMETER ACCESS FUNCTIONS
# =============================================================================

def get_instrumentation_parameter(param_name: str) -> Parameter:
    """Get a specific instrumentation parameter"""
    if param_name not in INSTRUMENTATION_PARAMETERS:
        raise KeyError(f"Parameter '{param_name}' not found in instrumentation registry")
    return INSTRUMENTATION_PARAMETERS[param_name]


def get_all_instrumentation_parameters() -> Dict[str, Parameter]:
    """Get all instrumentation parameters"""
    return INSTRUMENTATION_PARAMETERS.copy()


def get_parameters_by_category(category: str) -> Dict[str, Parameter]:
    """
    Get all parameters in a category.

    Categories: 'doubling', 'voicing', 'dynamics', 'selection', 'register', 'technique'
    """
    prefix = f"instrumentation.{category}."
    return {
        name: param for name, param in INSTRUMENTATION_PARAMETERS.items()
        if name.startswith(prefix)
    }


def get_default_values() -> Dict[str, Any]:
    """Get dictionary of all default values"""
    return {
        name: param.default
        for name, param in INSTRUMENTATION_PARAMETERS.items()
    }


def validate_parameter_value(param_name: str, value: Any) -> bool:
    """Validate that a value is valid for a parameter"""
    param = get_instrumentation_parameter(param_name)

    if param.type == ParameterType.CONTINUOUS:
        return param.range[0] <= value <= param.range[1]
    elif param.type == ParameterType.DISCRETE:
        return param.range[0] <= value <= param.range[1] and isinstance(value, int)
    elif param.type == ParameterType.CATEGORICAL:
        return value in param.options
    elif param.type == ParameterType.BOOLEAN:
        return isinstance(value, bool)

    return False


# =============================================================================
# STATISTICS & REPORTING
# =============================================================================

def get_parameter_statistics() -> Dict[str, Any]:
    """Get statistics about instrumentation parameters"""
    total = len(INSTRUMENTATION_PARAMETERS)

    by_category = {}
    for name in INSTRUMENTATION_PARAMETERS:
        category = name.split('.')[1]  # e.g., 'doubling' from 'instrumentation.doubling.xxx'
        by_category[category] = by_category.get(category, 0) + 1

    by_type = {}
    for param in INSTRUMENTATION_PARAMETERS.values():
        ptype = param.type.value
        by_type[ptype] = by_type.get(ptype, 0) + 1

    return {
        'total_parameters': total,
        'by_category': by_category,
        'by_type': by_type,
        'target': 50,
        'completion': f"{total}/50 ({total/50*100:.0f}%)"
    }


if __name__ == "__main__":
    # Print parameter statistics
    print("=" * 80)
    print("INSTRUMENTATION PARAMETERS - Agent 7")
    print("=" * 80)

    stats = get_parameter_statistics()
    print(f"\nTotal Parameters: {stats['total_parameters']}/{stats['target']}")
    print(f"Completion: {stats['completion']}")

    print("\nBy Category:")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"  {cat}: {count}")

    print("\nBy Type:")
    for ptype, count in sorted(stats['by_type'].items()):
        print(f"  {ptype}: {count}")

    # Show sample parameters
    print("\nSample Parameters:")
    samples = [
        "instrumentation.doubling.octave_probability",
        "instrumentation.voicing.close_position_ratio",
        "instrumentation.dynamics.balance_melody_ratio",
        "instrumentation.selection.orchestration_size"
    ]

    for param_name in samples:
        param = get_instrumentation_parameter(param_name)
        print(f"\n{param.name}:")
        print(f"  Type: {param.type.value}")
        print(f"  Default: {param.default}")
        if param.range:
            print(f"  Range: {param.range}")
        if param.options:
            print(f"  Options: {param.options}")
        print(f"  Description: {param.description}")

    print("\n" + "=" * 80)
    print("Agent 7 parameter definition complete!")
    print("=" * 80)
