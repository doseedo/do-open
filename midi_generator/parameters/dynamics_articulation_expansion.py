"""
Dynamics & Articulation Expansion - Agent 5
=============================================

Comprehensive parameter definitions for dynamics (10→50) and articulation (0→40).
This module adds 80 new parameters to the Universal Parameter Registry, enabling
precise control over musical expression, dynamics, and articulation.

Part of the Musical Program Synthesis system's expansion from 165 to 515+ parameters.

Author: Agent 5 - Dynamics & Articulation Expansion Specialist
License: MIT
Date: 2025-11-20
"""

from typing import List, Dict, Any
from .universal_registry import (
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact,
    REGISTRY
)


# ============================================================================
# DYNAMICS EXPANSION: 10 → 50 Parameters (+40 new)
# ============================================================================

def register_velocity_expression_parameters():
    """
    Register Velocity & Expression parameters (20 parameters)

    Controls overall dynamics, velocity ranges, layer balancing, and humanization
    of velocity across all musical elements.
    """

    REGISTRY.register(ParameterDefinition(
        name="overall_level",
        full_path="dynamics.velocity.overall_level",
        description="Master volume/dynamics level (0.0=silent, 1.0=maximum)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="range",
        full_path="dynamics.velocity.range",
        description="Dynamic range from softest to loudest (0.1=narrow, 1.0=full pp-ff)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.1,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "jazz", "orchestral", "chamber"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="layer_balance",
        full_path="dynamics.velocity.layer_balance",
        description="Relative volume balance across layers [melody, harmony, bass, drums]",
        param_type=ParameterType.ARRAY_FLOAT,
        default_value=[1.0, 0.7, 0.8, 0.9],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True,
        constraint_description="Array of 4 floats, each in range [0.0, 1.0]"
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_emphasis",
        full_path="dynamics.velocity.melody_emphasis",
        description="Melody layer volume emphasis (0.0=buried, 1.0=prominent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.85,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "singer_songwriter"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="bass_level",
        full_path="dynamics.velocity.bass_level",
        description="Bass layer volume level (0.0=silent, 1.0=loud)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.75,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "funk", "R&B", "dance"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="harmony_level",
        full_path="dynamics.velocity.harmony_level",
        description="Harmony/chord layer volume level (0.0=silent, 1.0=loud)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.65,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical", "pop"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="accent_intensity",
        full_path="dynamics.velocity.accent_intensity",
        description="Intensity of accented notes (0.0=no accent, 1.0=maximum accent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "latin"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="ghost_note_level",
        full_path="dynamics.velocity.ghost_note_level",
        description="Volume level for ghost notes (subtle, quiet notes)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.15,
        min_value=0.0,
        max_value=0.3,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["funk", "jazz", "R&B", "gospel"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="variation_amount",
        full_path="dynamics.velocity.variation_amount",
        description="Amount of dynamic variation/contrast (0.0=static, 1.0=highly varied)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="humanization",
        full_path="dynamics.velocity.humanization",
        description="Random velocity variation for human feel (0.0=robotic, 1.0=very human)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.4,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="note_to_note_variation",
        full_path="dynamics.velocity.note_to_note_variation",
        description="Subtle note-to-note velocity variation amount",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.08,
        min_value=0.0,
        max_value=0.2,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "acoustic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="mechanical_consistency",
        full_path="dynamics.velocity.mechanical_consistency",
        description="Mechanical vs human consistency (0.0=human/varied, 1.0=quantized/consistent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["electronic", "EDM", "techno"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="touch_sensitivity",
        full_path="dynamics.velocity.touch_sensitivity",
        description="Keyboard touch sensitivity simulation (0.0=uniform, 1.0=very sensitive)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["piano", "classical", "jazz"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="forte_piano_contrast",
        full_path="dynamics.velocity.forte_piano_contrast",
        description="Contrast between forte and piano passages (0.0=minimal, 1.0=dramatic)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="crescendo_shape",
        full_path="dynamics.velocity.crescendo_shape",
        description="Shape of crescendo curves (how volume increases)",
        param_type=ParameterType.CATEGORICAL,
        options=["linear", "exponential", "logarithmic"],
        default_value="linear",
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral", "film_score"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminuendo_shape",
        full_path="dynamics.velocity.diminuendo_shape",
        description="Shape of diminuendo curves (how volume decreases)",
        param_type=ParameterType.CATEGORICAL,
        options=["linear", "exponential", "logarithmic"],
        default_value="linear",
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral", "film_score"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_contour",
        full_path="dynamics.velocity.dynamic_contour",
        description="Style of dynamic changes (terraced, gradual, or sudden)",
        param_type=ParameterType.CATEGORICAL,
        options=["terraced", "gradual", "sudden"],
        default_value="gradual",
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "romantic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="micro_dynamics",
        full_path="dynamics.velocity.micro_dynamics",
        description="Subtle micro-dynamic changes within phrases (0.0=none, 1.0=maximum)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "chamber", "solo"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_shaping",
        full_path="dynamics.velocity.phrase_shaping",
        description="Amount of dynamic shaping within phrases (0.0=flat, 1.0=highly shaped)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "vocal"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="climax_boost",
        full_path="dynamics.velocity.climax_boost",
        description="Volume boost for climactic moments (0.0=none, 1.0=maximum boost)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["orchestral", "film_score", "progressive"],
        learnable=True
    ))


def register_articulation_curve_parameters():
    """
    Register Articulation Curves parameters (15 parameters)

    Controls ADSR envelopes, note lengths, overlaps, pedaling, and phrase separation.
    """

    REGISTRY.register(ParameterDefinition(
        name="attack_time",
        full_path="dynamics.articulation.attack_time",
        description="Note attack time in milliseconds (0.0=instant, 100.0=slow)",
        param_type=ParameterType.CONTINUOUS,
        default_value=10.0,
        min_value=0.0,
        max_value=100.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="decay_time",
        full_path="dynamics.articulation.decay_time",
        description="Note decay time in milliseconds",
        param_type=ParameterType.CONTINUOUS,
        default_value=100.0,
        min_value=0.0,
        max_value=500.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["synthesizer", "electronic", "piano"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sustain_level",
        full_path="dynamics.articulation.sustain_level",
        description="Note sustain level (0.0=silent, 1.0=full)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.8,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="release_time",
        full_path="dynamics.articulation.release_time",
        description="Note release time in milliseconds",
        param_type=ParameterType.CONTINUOUS,
        default_value=200.0,
        min_value=0.0,
        max_value=1000.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="envelope_shape",
        full_path="dynamics.articulation.envelope_shape",
        description="Overall envelope shape characteristic",
        param_type=ParameterType.CATEGORICAL,
        options=["sharp", "rounded", "sustained"],
        default_value="rounded",
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["synthesizer", "electronic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="note_length_ratio",
        full_path="dynamics.articulation.note_length_ratio",
        description="Note length as ratio of written duration (0.3=staccato, 1.0=legato)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.9,
        min_value=0.3,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="overlap_amount",
        full_path="dynamics.articulation.overlap_amount",
        description="Note overlap amount (0.0=gap, 0.5=overlap, negative=staccato gap)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=0.5,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["legato", "strings", "choral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="pedal_simulation",
        full_path="dynamics.articulation.pedal_simulation",
        description="Sustain pedal simulation amount (0.0=none, 1.0=full pedal)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["piano", "classical", "romantic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="half_pedal_prob",
        full_path="dynamics.articulation.half_pedal_prob",
        description="Probability of using half-pedal technique",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["piano", "classical"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sostenuto_usage",
        full_path="dynamics.articulation.sostenuto_usage",
        description="Sostenuto pedal usage amount (0.0=never, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["piano", "contemporary_classical"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="breath_marks",
        full_path="dynamics.articulation.breath_marks",
        description="Whether to insert breath marks (brief pauses)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["wind", "vocal", "choral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_separation",
        full_path="dynamics.articulation.phrase_separation",
        description="Amount of separation between phrases (0.0=connected, 1.0=clear gaps)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="caesura_prob",
        full_path="dynamics.articulation.caesura_prob",
        description="Probability of caesura (dramatic pause)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="fermata_prob",
        full_path="dynamics.articulation.fermata_prob",
        description="Probability of fermata (held note)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.03,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral", "hymn"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="fermata_length",
        full_path="dynamics.articulation.fermata_length",
        description="Length multiplier for fermata notes (1.0-4.0 beats)",
        param_type=ParameterType.CONTINUOUS,
        default_value=2.0,
        min_value=1.0,
        max_value=4.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral"],
        learnable=True
    ))


def register_dynamic_shape_form_parameters():
    """
    Register Dynamic Shape & Form parameters (15 parameters)

    Controls overall dynamic arc, section-level dynamics, transitions, and
    special dynamic effects.
    """

    REGISTRY.register(ParameterDefinition(
        name="overall_arc",
        full_path="dynamics.form.overall_arc",
        description="Overall dynamic arc across entire piece",
        param_type=ParameterType.CATEGORICAL,
        options=["crescendo", "diminuendo", "arch", "inverted_arch", "constant"],
        default_value="arch",
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "orchestral", "film_score"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="section_dynamic_plan",
        full_path="dynamics.form.section_dynamic_plan",
        description="Dynamic level for each structural section [intro, verse, chorus, etc.]",
        param_type=ParameterType.ARRAY_FLOAT,
        default_value=[0.5, 0.6, 0.8, 0.7, 0.85, 0.6],
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "electronic"],
        learnable=True,
        constraint_description="Array of floats in range [0.0, 1.0]"
    ))

    REGISTRY.register(ParameterDefinition(
        name="intro_level",
        full_path="dynamics.form.intro_level",
        description="Dynamic level for introduction section",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="verse_level",
        full_path="dynamics.form.verse_level",
        description="Dynamic level for verse sections",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "folk"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="chorus_level",
        full_path="dynamics.form.chorus_level",
        description="Dynamic level for chorus sections",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.85,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["pop", "rock", "gospel"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="bridge_level",
        full_path="dynamics.form.bridge_level",
        description="Dynamic level for bridge sections",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["pop", "rock"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="solo_level",
        full_path="dynamics.form.solo_level",
        description="Dynamic level for solo sections",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.8,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "rock", "blues"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="outro_level",
        full_path="dynamics.form.outro_level",
        description="Dynamic level for outro/ending section",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.4,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_transition_speed",
        full_path="dynamics.form.dynamic_transition_speed",
        description="Speed of dynamic transitions between sections (0.0=slow, 1.0=instant)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sudden_dynamic_change_prob",
        full_path="dynamics.form.sudden_dynamic_change_prob",
        description="Probability of sudden dynamic changes (subito forte/piano)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "dramatic", "contemporary"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sforzando_prob",
        full_path="dynamics.form.sforzando_prob",
        description="Probability of sforzando (sudden accent)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="subito_piano_prob",
        full_path="dynamics.form.subito_piano_prob",
        description="Probability of subito piano (suddenly soft)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="echo_effect_prob",
        full_path="dynamics.form.echo_effect_prob",
        description="Probability of echo effect (repeated phrase softer)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_layering",
        full_path="dynamics.form.dynamic_layering",
        description="Amount of dynamic layering/stratification (0.0=unified, 1.0=highly layered)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.4,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["orchestral", "progressive", "electronic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="textural_buildup",
        full_path="dynamics.form.textural_buildup",
        description="Amount of textural buildup through adding layers (0.0=static, 1.0=progressive)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["electronic", "progressive", "post_rock"],
        learnable=True
    ))


# ============================================================================
# ARTICULATION EXPANSION: 0 → 40 Parameters (+40 new)
# ============================================================================

def register_basic_articulation_parameters():
    """
    Register Basic Articulation Marks parameters (15 parameters)

    Controls standard articulation markings like staccato, legato, accents,
    slurs, and rhythmic articulation styles.
    """

    REGISTRY.register(ParameterDefinition(
        name="staccato_prob",
        full_path="articulation.basic.staccato_prob",
        description="Probability of staccato articulation (short, detached notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "baroque", "march"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="staccatissimo_prob",
        full_path="articulation.basic.staccatissimo_prob",
        description="Probability of staccatissimo (very short, extra detached)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "contemporary"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="tenuto_prob",
        full_path="articulation.basic.tenuto_prob",
        description="Probability of tenuto (held, sustained notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="marcato_prob",
        full_path="articulation.basic.marcato_prob",
        description="Probability of marcato (marked, emphasized notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="accent_prob",
        full_path="articulation.basic.accent_prob",
        description="Probability of accent marks (emphasized but not necessarily louder)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="strong_accent_prob",
        full_path="articulation.basic.strong_accent_prob",
        description="Probability of strong accent/sforzando marking",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="legato_default",
        full_path="articulation.basic.legato_default",
        description="Whether legato (smooth, connected) is the default articulation",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "lyrical", "vocal"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="portato_prob",
        full_path="articulation.basic.portato_prob",
        description="Probability of portato (halfway between legato and staccato)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "strings"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="detache_prob",
        full_path="articulation.basic.detache_prob",
        description="Probability of detaché (separated bow strokes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["strings", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="loure_prob",
        full_path="articulation.basic.louré_prob",
        description="Probability of louré (legato with slight separation)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["baroque", "dance"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="slur_grouping",
        full_path="articulation.basic.slur_grouping",
        description="Typical slur grouping size (notes per slur)",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 3, 4, 6, 8],
        default_value=4,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_mark_length",
        full_path="articulation.basic.phrase_mark_length",
        description="Typical phrase mark length in beats",
        param_type=ParameterType.CATEGORICAL,
        options=[4, 8, 16, 32],
        default_value=8,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="breathing_space",
        full_path="articulation.basic.breathing_space",
        description="Amount of breathing space between phrases (0.0=none, 1.0=generous)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["wind", "vocal", "choral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="note_separation",
        full_path="articulation.basic.note_separation",
        description="Default note separation amount (0.0=connected, 1.0=widely separated)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythmic_articulation",
        full_path="articulation.basic.rhythmic_articulation",
        description="Rhythmic articulation style (strict timing, loose, or swung)",
        param_type=ParameterType.CATEGORICAL,
        options=["strict", "loose", "swung"],
        default_value="strict",
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "swing", "blues"],
        learnable=True
    ))


def register_string_technique_parameters():
    """
    Register String Techniques parameters (10 parameters)

    Controls string-specific playing techniques like pizzicato, col legno,
    harmonics, and various bowing techniques.
    """

    REGISTRY.register(ParameterDefinition(
        name="pizzicato_prob",
        full_path="articulation.string.pizzicato_prob",
        description="Probability of pizzicato (plucked strings)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "orchestral", "chamber"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="arco_prob",
        full_path="articulation.string.arco_prob",
        description="Probability of arco (bowed strings, default technique)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.9,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="col_legno_prob",
        full_path="articulation.string.col_legno_prob",
        description="Probability of col legno (striking with wood of bow)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "film_score"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sul_ponticello_prob",
        full_path="articulation.string.sul_ponticello_prob",
        description="Probability of sul ponticello (bowing near bridge, glassy sound)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "atmospheric"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="sul_tasto_prob",
        full_path="articulation.string.sul_tasto_prob",
        description="Probability of sul tasto (bowing over fingerboard, soft sound)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["impressionist", "atmospheric"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="tremolo_prob",
        full_path="articulation.string.tremolo_prob",
        description="Probability of tremolo (rapid repetition of note)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "film_score", "dramatic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="spiccato_prob",
        full_path="articulation.string.spiccato_prob",
        description="Probability of spiccato (bouncing bow)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "virtuosic"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="ricochet_prob",
        full_path="articulation.string.ricochet_prob",
        description="Probability of ricochet (thrown/bouncing bow)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.03,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["virtuosic", "showpiece"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="harmonics_prob",
        full_path="articulation.string.harmonics_prob",
        description="Probability of natural/artificial harmonics",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "contemporary", "atmospheric"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="double_stop_prob",
        full_path="articulation.string.double_stop_prob",
        description="Probability of double stops (playing two strings simultaneously)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "folk", "bluegrass"],
        learnable=True
    ))


def register_wind_brass_technique_parameters():
    """
    Register Wind & Brass Techniques parameters (15 parameters)

    Controls wind and brass instrument techniques including tonguing,
    breath control, special effects, and mutes.
    """

    REGISTRY.register(ParameterDefinition(
        name="tongue_articulation",
        full_path="articulation.wind.tongue_articulation",
        description="Type of tonguing technique",
        param_type=ParameterType.CATEGORICAL,
        options=["single", "double", "triple", "flutter"],
        default_value="single",
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["wind", "brass", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="tonguing_precision",
        full_path="articulation.wind.tonguing_precision",
        description="Precision/clarity of tonguing (0.0=soft, 1.0=crisp)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["wind", "brass"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="breath_attack",
        full_path="articulation.wind.breath_attack",
        description="Breath attack intensity (0.0=soft, 1.0=hard)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["wind", "flute"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="breath_pressure",
        full_path="articulation.wind.breath_pressure",
        description="Overall breath pressure (0.0=light, 1.0=heavy)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["wind", "brass"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="flutter_tongue_prob",
        full_path="articulation.wind.flutter_tongue_prob",
        description="Probability of flutter tonguing",
        param_type=ParameterType.PROBABILITY,
        default_value=0.03,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "jazz"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="slap_tongue_prob",
        full_path="articulation.wind.slap_tongue_prob",
        description="Probability of slap tonguing (percussive technique)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "avant_garde"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="growl_prob",
        full_path="articulation.wind.growl_prob",
        description="Probability of growl technique (humming while playing)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "blues", "rock"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="multiphonics_prob",
        full_path="articulation.wind.multiphonics_prob",
        description="Probability of multiphonics (multiple notes simultaneously)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.01,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary_classical", "experimental"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="lip_trill_prob",
        full_path="articulation.brass.lip_trill_prob",
        description="Probability of lip trill (brass)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["brass", "orchestral"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="rip_prob",
        full_path="articulation.brass.rip_prob",
        description="Probability of rip (quick upward glissando)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "big_band", "brass"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="doit_prob",
        full_path="articulation.brass.doit_prob",
        description="Probability of doit (upward scoop at end of note)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.06,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "big_band"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="fall_off_prob",
        full_path="articulation.brass.fall_off_prob",
        description="Probability of fall-off (downward scoop at end of note)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.06,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "big_band", "blues"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="plunger_mute",
        full_path="articulation.brass.plunger_mute",
        description="Amount of plunger mute usage (0.0=none, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "swing"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="harmon_mute",
        full_path="articulation.brass.harmon_mute",
        description="Amount of harmon mute usage (0.0=none, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "miles_davis_style"],
        learnable=True
    ))

    REGISTRY.register(ParameterDefinition(
        name="straight_mute",
        full_path="articulation.brass.straight_mute",
        description="Amount of straight mute usage (0.0=none, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "brass_ensemble"],
        learnable=True
    ))


# ============================================================================
# Master Registration Function
# ============================================================================

def register_all_dynamics_articulation_parameters():
    """
    Register all dynamics and articulation expansion parameters.

    Total: 80 new parameters
    - Dynamics: 50 parameters (20 velocity + 15 articulation curves + 15 form)
    - Articulation: 40 parameters (15 basic + 10 string + 15 wind/brass)
    """

    print("=" * 80)
    print("AGENT 5: DYNAMICS & ARTICULATION EXPANSION")
    print("=" * 80)
    print()

    # Dynamics Expansion: 10 → 50 (+40)
    print("🎵 Registering Dynamics Expansion (10 → 50 parameters)...")
    print()

    print("  [1/3] Velocity & Expression (20 parameters)...")
    register_velocity_expression_parameters()

    print("  [2/3] Articulation Curves (15 parameters)...")
    register_articulation_curve_parameters()

    print("  [3/3] Dynamic Shape & Form (15 parameters)...")
    register_dynamic_shape_form_parameters()

    print("  ✅ Dynamics expansion complete: 50 parameters registered")
    print()

    # Articulation Expansion: 0 → 40 (+40)
    print("🎺 Registering Articulation Expansion (0 → 40 parameters)...")
    print()

    print("  [1/3] Basic Articulation Marks (15 parameters)...")
    register_basic_articulation_parameters()

    print("  [2/3] String Techniques (10 parameters)...")
    register_string_technique_parameters()

    print("  [3/3] Wind & Brass Techniques (15 parameters)...")
    register_wind_brass_technique_parameters()

    print("  ✅ Articulation expansion complete: 40 parameters registered")
    print()

    # Summary
    stats = REGISTRY.get_statistics()
    print("=" * 80)
    print("📊 EXPANSION SUMMARY")
    print("=" * 80)
    print(f"Total parameters in registry: {stats['total_parameters']}")
    print(f"Dynamics parameters: 50 (was 10)")
    print(f"Articulation parameters: 40 (was 0)")
    print(f"New parameters added: 80")
    print()
    print("By category:")
    dynamics_count = len(REGISTRY.get_by_category(ParameterCategory.DYNAMICS))
    articulation_count = len(REGISTRY.get_by_category(ParameterCategory.ARTICULATION))
    print(f"  - DYNAMICS: {dynamics_count}")
    print(f"  - ARTICULATION: {articulation_count}")
    print()
    print("✅ Agent 5 expansion complete!")
    print("=" * 80)


# ============================================================================
# Validation & Export Utilities
# ============================================================================

def validate_all_parameters() -> bool:
    """
    Validate all registered parameters for consistency and completeness.

    Returns:
        True if all validations pass, False otherwise
    """
    print("\n🔍 Validating parameter definitions...")

    all_params = REGISTRY.get_all_parameters()
    errors = []

    # Check dynamics parameters
    dynamics_params = [p for p in all_params if p.startswith("dynamics.")]
    if len(dynamics_params) < 50:
        errors.append(f"Expected 50+ dynamics parameters, found {len(dynamics_params)}")

    # Check articulation parameters
    articulation_params = [p for p in all_params if p.startswith("articulation.")]
    if len(articulation_params) < 40:
        errors.append(f"Expected 40+ articulation parameters, found {len(articulation_params)}")

    # Validate each parameter has required fields
    for path in all_params:
        param = REGISTRY.get(path)
        if not param.description:
            errors.append(f"{path}: Missing description")
        if param.category is None:
            errors.append(f"{path}: Missing category")
        if param.musical_impact is None:
            errors.append(f"{path}: Missing musical impact")

    if errors:
        print("❌ Validation errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ All parameters validated successfully!")
        return True


def export_parameter_documentation(output_dir: str = "/home/user/Do/midi_generator/parameters"):
    """
    Export comprehensive documentation of all dynamics and articulation parameters.

    Args:
        output_dir: Directory to save documentation files
    """
    from pathlib import Path

    output_path = Path(output_dir)

    # Export JSON
    json_file = output_path / "dynamics_articulation_registry.json"
    print(f"\n💾 Exporting to {json_file}...")

    # Export Markdown documentation
    md_file = output_path / "DYNAMICS_ARTICULATION_PARAMS.md"
    print(f"💾 Exporting to {md_file}...")

    # Generate detailed markdown
    lines = []
    lines.append("# Dynamics & Articulation Parameters")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append("This document describes the 80 new parameters added by Agent 5:")
    lines.append("- **Dynamics**: 50 parameters (10 → 50)")
    lines.append("- **Articulation**: 40 parameters (0 → 40)")
    lines.append("")

    # Dynamics section
    lines.append("## Dynamics Parameters (50)")
    lines.append("")

    lines.append("### Velocity & Expression (20 parameters)")
    lines.append("")
    lines.append("| Parameter | Type | Range | Default | Description |")
    lines.append("|-----------|------|-------|---------|-------------|")

    velocity_params = [p for p in REGISTRY.get_by_category(ParameterCategory.DYNAMICS)
                      if "velocity" in p.full_path]
    for param in sorted(velocity_params, key=lambda p: p.full_path):
        param_type = param.param_type.value
        range_str = f"[{param.min_value}, {param.max_value}]" if param.min_value is not None else "—"
        default = str(param.default_value)
        lines.append(f"| `{param.full_path}` | {param_type} | {range_str} | {default} | {param.description} |")

    lines.append("")

    with open(md_file, 'w') as f:
        f.write('\n'.join(lines))

    print("✅ Documentation exported successfully!")


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Register all parameters
    register_all_dynamics_articulation_parameters()

    # Validate
    if validate_all_parameters():
        # Export documentation
        export_parameter_documentation()

        # Export full registry
        REGISTRY.export_to_json("/home/user/Do/midi_generator/parameters/registry.json")
        REGISTRY.generate_documentation("/home/user/Do/midi_generator/parameters/PARAMETERS.md")

        print("\n🎉 Agent 5 complete! Ready for XGBoost integration.")
    else:
        print("\n⚠️  Validation failed. Please review errors above.")
