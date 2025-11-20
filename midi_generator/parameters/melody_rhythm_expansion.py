"""
Melody & Rhythm Expansion - Agent 4
===================================

Comprehensive parameter definitions for melody and rhythm expansion
This module adds 120 new parameters to the Universal Parameter Registry:
- 63 new melody parameters (37 → 100)
- 57 new rhythm parameters (43 → 100)

Expansion Strategy:
- Melodic Contour & Shape: 20 parameters
- Interval & Chromaticism: 23 parameters
- Ornamentation & Articulation: 20 parameters
- Advanced Rhythmic Patterns: 20 parameters
- Syncopation & Feel: 22 parameters
- Density, Complexity & Fills: 15 parameters

Total New Parameters: 120
Target Lines: 3000+

Author: Agent 4 - Melody & Rhythm Expansion Specialist
License: MIT
"""

from .universal_registry import (
    ParameterDefinition, ParameterType, ParameterCategory,
    MusicalImpact, REGISTRY
)


def register_melody_expansion_parameters():
    """
    Register 63 new melody parameters (37 → 100)

    Categories:
    1. Melodic Contour & Shape (20 params)
    2. Interval & Chromaticism (23 params)
    3. Ornamentation & Articulation (20 params)
    """

    # ========================================================================
    # MELODIC CONTOUR & SHAPE PARAMETERS (20 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_arch_probability",
        full_path="melody.contour.arch_probability",
        description="Probability of arch-shaped melodic contour (rise then fall)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "film_score", "folk"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_inverted_arch_prob",
        full_path="melody.contour.inverted_arch_prob",
        description="Probability of inverted arch contour (fall then rise)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_ascending_prob",
        full_path="melody.contour.ascending_prob",
        description="Probability of predominantly ascending melodic line",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "film_score", "anthem"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_descending_prob",
        full_path="melody.contour.descending_prob",
        description="Probability of predominantly descending melodic line",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "folk", "ballad"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_wave_pattern_prob",
        full_path="melody.contour.wave_pattern_prob",
        description="Probability of wave-like melodic patterns (continuous rise/fall)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["impressionist", "ambient", "new_age"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_terraced_dynamics",
        full_path="melody.contour.terraced_dynamics",
        description="Enable terraced dynamics (stepwise changes in register/dynamics)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["baroque", "minimalist"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_climax_placement",
        full_path="melody.contour.climax_placement",
        description="Position of melodic climax (0.0=beginning, 1.0=end)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.618,  # Golden ratio
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_apex_note_emphasis",
        full_path="melody.contour.apex_note_emphasis",
        description="Degree of emphasis on the highest melodic note (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "opera", "anthem"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_range_exploration",
        full_path="melody.contour.range_exploration",
        description="How much of the available melodic range to explore (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["virtuosic", "contemporary", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_register_shifts",
        full_path="melody.contour.register_shifts",
        description="Frequency of sudden register shifts/leaps (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.2,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary", "jazz", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_contour_octave_displacement_prob",
        full_path="melody.contour.octave_displacement_prob",
        description="Probability of octave displacement of melodic notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["baroque", "contemporary", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_phrase_peak_placement",
        full_path="melody.shape.phrase_peak_placement",
        description="Where melodic peaks occur within phrases",
        param_type=ParameterType.CATEGORICAL,
        options=["early", "middle", "late", "multiple"],
        default_value="middle",
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_melodic_curve_smoothness",
        full_path="melody.shape.melodic_curve_smoothness",
        description="How smooth vs. angular the melodic curve is (0.0=angular, 1.0=smooth)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["legato", "cantabile", "lyrical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_directional_consistency",
        full_path="melody.shape.directional_consistency",
        description="How consistently the melody maintains direction (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["minimalist", "meditative"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_gap_fill_principle",
        full_path="melody.shape.gap_fill_principle",
        description="Tendency for leaps to be followed by stepwise motion in opposite direction (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "renaissance", "folk"],
        constraint_description="Classic counterpoint principle"
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_registral_direction",
        full_path="melody.shape.registral_direction",
        description="Overall registral tendency of the melody",
        param_type=ParameterType.CATEGORICAL,
        options=["balanced", "rising", "falling"],
        default_value="balanced",
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_tessitura",
        full_path="melody.shape.tessitura",
        description="The general register/range where the melody sits",
        param_type=ParameterType.CATEGORICAL,
        options=["low", "mid", "high", "wide"],
        default_value="mid",
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        constraint_description="Vocal range consideration"
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_melodic_peaks_frequency",
        full_path="melody.shape.melodic_peaks_frequency",
        description="Frequency of melodic high points (0.0=rare, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["dramatic", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_valley_frequency",
        full_path="melody.shape.valley_frequency",
        description="Frequency of melodic low points (0.0=rare, 1.0=frequent)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.25,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["blues", "folk"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_shape_contour_complexity",
        full_path="melody.shape.contour_complexity",
        description="Overall complexity of melodic contour (1=simple, 7=highly complex)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    # ========================================================================
    # INTERVAL & CHROMATICISM PARAMETERS (23 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_stepwise_motion_prob",
        full_path="melody.interval.stepwise_motion_prob",
        description="Probability of stepwise melodic motion (seconds)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.65,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["folk", "classical", "vocal"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_leap_probability",
        full_path="melody.interval.leap_probability",
        description="Probability of melodic leaps (intervals > major 2nd)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.35,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_max_leap_interval",
        full_path="melody.interval.max_leap_interval",
        description="Maximum melodic leap interval in semitones",
        param_type=ParameterType.CATEGORICAL,
        options=[3, 4, 5, 7, 8, 12, 15, 19, 24],
        default_value=12,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_leap_resolution",
        full_path="melody.interval.leap_resolution",
        description="Require leaps to be followed by motion in opposite direction",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "renaissance", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_chromatic_approach",
        full_path="melody.interval.chromatic_approach",
        description="Probability of chromatic approach to target notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "chromatic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_chromatic_passing_tone",
        full_path="melody.interval.chromatic_passing_tone",
        description="Probability of chromatic passing tones between diatonic notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "romantic", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_chromatic_neighbor_tone",
        full_path="melody.interval.chromatic_neighbor_tone",
        description="Probability of chromatic neighbor tones (chromatic embellishment)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "baroque", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_chromatic_enclosure",
        full_path="melody.interval.chromatic_enclosure",
        description="Probability of chromatic enclosure (upper & lower chromatic approach)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "bebop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_double_chromatic_approach",
        full_path="melody.interval.double_chromatic_approach",
        description="Probability of double chromatic approach (two chromatic notes before target)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["bebop", "modern_jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_bebop_chromatic_prob",
        full_path="melody.interval.bebop_chromatic_prob",
        description="Probability of bebop-style chromatic passing tones on upbeats",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["bebop", "jazz", "swing"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_diatonic_passing_tone",
        full_path="melody.interval.diatonic_passing_tone",
        description="Probability of diatonic passing tones",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_diatonic_neighbor",
        full_path="melody.interval.diatonic_neighbor",
        description="Probability of diatonic neighbor tones",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "folk", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_anticipation_prob",
        full_path="melody.interval.anticipation_prob",
        description="Probability of anticipation (early arrival on next chord tone)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "pop", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_escape_tone_prob",
        full_path="melody.interval.escape_tone_prob",
        description="Probability of escape tones (stepwise approach, leap away)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["classical", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_appoggiatura_prob",
        full_path="melody.interval.appoggiatura_prob",
        description="Probability of appoggiaturas (accented non-chord tones)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_cambiata_prob",
        full_path="melody.interval.cambiata_prob",
        description="Probability of cambiata patterns (step-leap-step in same direction)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["renaissance", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_tritone_usage",
        full_path="melody.interval.tritone_usage",
        description="Probability of melodic tritone intervals",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary", "jazz", "avant_garde"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_augmented_interval_prob",
        full_path="melody.interval.augmented_interval_prob",
        description="Probability of augmented intervals in melody",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "jazz", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_diminished_interval_prob",
        full_path="melody.interval.diminished_interval_prob",
        description="Probability of diminished intervals in melody",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz", "avant_garde"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_compound_interval_prob",
        full_path="melody.interval.compound_interval_prob",
        description="Probability of compound intervals (> octave)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary", "virtuosic", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_parallel_intervals",
        full_path="melody.interval.parallel_intervals",
        description="Type of parallel intervals to use in multi-voice melody",
        param_type=ParameterType.CATEGORICAL,
        options=["none", "thirds", "sixths", "fourths", "fifths", "octaves"],
        default_value="none",
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["folk", "blues", "gregorian"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_consonance_preference",
        full_path="melody.interval.consonance_preference",
        description="Preference for consonant intervals (0.0=dissonant, 1.0=consonant)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_interval_tension_resolution_cycle",
        full_path="melody.interval.tension_resolution_cycle",
        description="How often to cycle through tension and resolution in bars",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=16,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    # ========================================================================
    # ORNAMENTATION & ARTICULATION PARAMETERS (20 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_trill_probability",
        full_path="melody.ornament.trill_probability",
        description="Probability of adding trills to melodic notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["baroque", "classical", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_mordent_prob",
        full_path="melody.ornament.mordent_prob",
        description="Probability of mordent ornaments",
        param_type=ParameterType.PROBABILITY,
        default_value=0.03,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["baroque", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_turn_prob",
        full_path="melody.ornament.turn_prob",
        description="Probability of turn ornaments",
        param_type=ParameterType.PROBABILITY,
        default_value=0.04,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["baroque", "classical", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_grace_note_prob",
        full_path="melody.ornament.grace_note_prob",
        description="Probability of grace notes before main notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["baroque", "romantic", "jazz", "celtic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_grace_note_type",
        full_path="melody.ornament.grace_note_type",
        description="Type of grace note to use",
        param_type=ParameterType.CATEGORICAL,
        options=["acciaccatura", "appoggiatura", "both"],
        default_value="acciaccatura",
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["baroque", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_glissando_prob",
        full_path="melody.ornament.glissando_prob",
        description="Probability of glissando between notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "blues", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_portamento_prob",
        full_path="melody.ornament.portamento_prob",
        description="Probability of portamento (smooth slide between notes)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.03,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["romantic", "jazz", "vocal"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_vibrato_intensity",
        full_path="melody.ornament.vibrato_intensity",
        description="Intensity of vibrato on sustained notes (0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["romantic", "opera", "vocal"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_tremolo_prob",
        full_path="melody.ornament.tremolo_prob",
        description="Probability of tremolo on sustained notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["romantic", "contemporary", "orchestral"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_shake_prob",
        full_path="melody.ornament.shake_prob",
        description="Probability of shake ornament (extended trill)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.01,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["baroque", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_fall_off_prob",
        full_path="melody.ornament.fall_off_prob",
        description="Probability of fall-off at end of notes (jazz articulation)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "blues"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_doit_prob",
        full_path="melody.ornament.doit_prob",
        description="Probability of doit at end of notes (upward scoop)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "big_band"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_scoop_prob",
        full_path="melody.ornament.scoop_prob",
        description="Probability of scoop at beginning of notes (upward approach)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["jazz", "blues", "gospel"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_ornament_bend_prob",
        full_path="melody.ornament.bend_prob",
        description="Probability of pitch bends on notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["blues", "rock", "country"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_legato_prob",
        full_path="melody.articulation.legato_prob",
        description="Probability of legato articulation (smooth, connected)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "lyrical", "cantabile"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_staccato_prob",
        full_path="melody.articulation.staccato_prob",
        description="Probability of staccato articulation (short, detached)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_tenuto_prob",
        full_path="melody.articulation.tenuto_prob",
        description="Probability of tenuto articulation (held full value)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_marcato_prob",
        full_path="melody.articulation.marcato_prob",
        description="Probability of marcato articulation (strongly accented)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "dramatic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_accent_prob",
        full_path="melody.articulation.accent_prob",
        description="Probability of accents on melodic notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="melody_articulation_slur_length",
        full_path="melody.articulation.slur_length",
        description="Typical length of slurs in number of notes",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 3, 4, 6, 8, 12, 16],
        default_value=4,
        category=ParameterCategory.ARTICULATION,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["romantic", "lyrical"]
    ))


def register_rhythm_expansion_parameters():
    """
    Register 57 new rhythm parameters (43 → 100)

    Categories:
    1. Advanced Rhythmic Patterns (20 params)
    2. Syncopation & Feel (22 params)
    3. Density, Complexity & Fills (15 params)
    """

    # ========================================================================
    # ADVANCED RHYTHMIC PATTERNS (20 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_polyrhythm_active",
        full_path="rhythm.pattern.polyrhythm_active",
        description="Enable polyrhythmic patterns",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["african", "contemporary", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_polyrhythm_ratio",
        full_path="rhythm.pattern.polyrhythm_ratio",
        description="Polyrhythm ratio (e.g., 3:2 means 3 beats against 2)",
        param_type=ParameterType.CATEGORICAL,
        options=["3:2", "4:3", "5:4", "7:4", "3:4", "5:8", "7:8", "5:3"],
        default_value="3:2",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["african", "progressive", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_hemiola_prob",
        full_path="rhythm.pattern.hemiola_prob",
        description="Probability of hemiola patterns (3:2 rhythmic displacement)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "folk"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_metric_modulation",
        full_path="rhythm.pattern.metric_modulation",
        description="Enable metric modulation (tempo changes via rhythmic relationships)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary", "progressive", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_tempo_relationship",
        full_path="rhythm.pattern.tempo_relationship",
        description="Tempo relationship for metric modulation",
        param_type=ParameterType.CATEGORICAL,
        options=["2:3", "3:4", "4:5", "3:2", "4:3", "5:4"],
        default_value="2:3",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_cross_rhythm_prob",
        full_path="rhythm.pattern.cross_rhythm_prob",
        description="Probability of cross-rhythms (rhythms that cut across the beat)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "african", "latin"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_polymetric_layers",
        full_path="rhythm.pattern.polymetric_layers",
        description="Number of simultaneous polymetric layers",
        param_type=ParameterType.CATEGORICAL,
        options=[0, 2, 3, 4],
        default_value=0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary", "experimental", "african"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_clave_pattern",
        full_path="rhythm.pattern.clave_pattern",
        description="Type of clave pattern to use as rhythmic foundation",
        param_type=ParameterType.CATEGORICAL,
        options=["none", "son", "rumba", "bossa", "afro_cuban", "cascara"],
        default_value="none",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["latin", "salsa", "afro_cuban", "bossa_nova"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_timeline_pattern",
        full_path="rhythm.pattern.timeline_pattern",
        description="African timeline pattern type",
        param_type=ParameterType.CATEGORICAL,
        options=["none", "standard", "bembe", "soukous", "gahu", "kpanlogo"],
        default_value="none",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["african", "world", "afrobeat"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_odd_meter_grouping",
        full_path="rhythm.pattern.odd_meter_grouping",
        description="Grouping pattern for odd meters (e.g., [2,2,3] for 7/8)",
        param_type=ParameterType.ARRAY_INT,
        default_value=[2, 2, 3],
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["balkan", "progressive", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_additive_rhythm",
        full_path="rhythm.pattern.additive_rhythm",
        description="Use additive rhythm patterns (2+2+3 style groupings)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["balkan", "middle_eastern", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_divisive_rhythm",
        full_path="rhythm.pattern.divisive_rhythm",
        description="Use divisive rhythm patterns (even divisions)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["western", "classical", "pop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_isorhythm",
        full_path="rhythm.pattern.isorhythm",
        description="Enable isorhythmic patterns (repeating rhythm with changing pitches)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["medieval", "contemporary", "minimalist"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_talea_length",
        full_path="rhythm.pattern.talea_length",
        description="Length of rhythmic pattern (talea) in isorhythm",
        param_type=ParameterType.CATEGORICAL,
        options=[3, 5, 7, 11, 13, 17],
        default_value=5,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["medieval", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_color_length",
        full_path="rhythm.pattern.color_length",
        description="Length of pitch pattern (color) in isorhythm",
        param_type=ParameterType.CATEGORICAL,
        options=[4, 6, 8, 12, 16],
        default_value=8,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["medieval", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_rhythmic_diminution",
        full_path="rhythm.pattern.rhythmic_diminution",
        description="Probability of rhythmic diminution (speeding up rhythmic values)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_rhythmic_augmentation",
        full_path="rhythm.pattern.rhythmic_augmentation",
        description="Probability of rhythmic augmentation (slowing down rhythmic values)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_rhythmic_retrograde",
        full_path="rhythm.pattern.rhythmic_retrograde",
        description="Probability of rhythmic retrograde (playing rhythm backwards)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "serial", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_rhythmic_inversion",
        full_path="rhythm.pattern.rhythmic_inversion",
        description="Probability of rhythmic inversion (long becomes short, short becomes long)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_pattern_rhythmic_canon",
        full_path="rhythm.pattern.rhythmic_canon",
        description="Probability of rhythmic canon (rhythm repeated in different voice)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["baroque", "classical", "contemporary"]
    ))

    # ========================================================================
    # SYNCOPATION & FEEL PARAMETERS (22 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_anticipation_8th",
        full_path="rhythm.syncopation.anticipation_8th",
        description="Probability of 8th note anticipation (early arrival)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "pop", "funk"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_anticipation_16th",
        full_path="rhythm.syncopation.anticipation_16th",
        description="Probability of 16th note anticipation",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "r_and_b", "hip_hop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_delayed_attack",
        full_path="rhythm.syncopation.delayed_attack",
        description="Probability of delayed attack (late arrival)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "blues", "laid_back"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_offbeat_emphasis",
        full_path="rhythm.syncopation.offbeat_emphasis",
        description="Probability of emphasizing offbeat notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["reggae", "ska", "funk"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_backbeat_strength",
        full_path="rhythm.syncopation.backbeat_strength",
        description="Strength of backbeat emphasis (beats 2 and 4)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["rock", "pop", "funk", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_upbeat_emphasis",
        full_path="rhythm.syncopation.upbeat_emphasis",
        description="Probability of emphasizing upbeats (& of beats)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "disco", "dance"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_cross_beat_accent",
        full_path="rhythm.syncopation.cross_beat_accent",
        description="Probability of accents that cross the main beat",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "progressive", "afro_cuban"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_syncopation_composite_rhythm_complexity",
        full_path="rhythm.syncopation.composite_rhythm_complexity",
        description="Overall complexity of composite rhythm (1=simple, 7=complex)",
        param_type=ParameterType.INTEGER,
        default_value=3,
        min_value=1,
        max_value=7,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_swing_ratio",
        full_path="rhythm.feel.swing_ratio",
        description="Swing ratio (0.5=straight, 0.67=triplet swing)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.5,
        max_value=0.75,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "swing", "blues"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_shuffle_intensity",
        full_path="rhythm.feel.shuffle_intensity",
        description="Intensity of shuffle feel (0.0=none, 1.0=heavy shuffle)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "shuffle", "boogie"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_straight_8ths",
        full_path="rhythm.feel.straight_8ths",
        description="Use straight 8th notes (no swing)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["rock", "pop", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_half_time_feel",
        full_path="rhythm.feel.half_time_feel",
        description="Use half-time feel (backbeat on 3 instead of 2 and 4)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["hip_hop", "r_and_b", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_double_time_feel",
        full_path="rhythm.feel.double_time_feel",
        description="Use double-time feel (perceived tempo doubles)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["bebop", "jazz", "uptempo"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_laid_back_timing",
        full_path="rhythm.feel.laid_back_timing",
        description="Amount of laid-back timing (slight delay behind beat, 0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "jazz", "soul"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_rushing_tendency",
        full_path="rhythm.feel.rushing_tendency",
        description="Tendency to rush (play ahead of beat, 0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["punk", "aggressive", "energetic"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_rubato_intensity",
        full_path="rhythm.feel.rubato_intensity",
        description="Amount of rubato (flexible tempo, 0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "expressive", "ballad"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_agogic_accent",
        full_path="rhythm.feel.agogic_accent",
        description="Probability of agogic accents (emphasis via duration)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic", "expressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_micro_timing_variation",
        full_path="rhythm.feel.micro_timing_variation",
        description="Amount of micro-timing variation (subtle timing imperfections, 0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.05,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_human_timing_imperfection",
        full_path="rhythm.feel.human_timing_imperfection",
        description="Amount of human-like timing imperfection (0.0=perfect, 1.0=very human)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_groove_quantization",
        full_path="rhythm.feel.groove_quantization",
        description="Groove quantization level (0.0=loose/free, 1.0=tight/quantized)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.8,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_pocket_depth",
        full_path="rhythm.feel.pocket_depth",
        description="Depth of rhythmic pocket (0.0=shallow, 1.0=deep groove)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "soul", "jazz", "groove"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_feel_subdivision_clarity",
        full_path="rhythm.feel.subdivision_clarity",
        description="Clarity of rhythmic subdivisions (0.0=implied, 1.0=explicit)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    # ========================================================================
    # DENSITY, COMPLEXITY & FILLS PARAMETERS (15 parameters)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="rhythm_density_note_density",
        full_path="rhythm.density.note_density",
        description="Overall note density (notes per bar, 0.0=sparse, 1.0=dense)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_density_rhythmic_activity_level",
        full_path="rhythm.density.rhythmic_activity_level",
        description="Level of rhythmic activity (1=minimal, 7=maximal)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_density_rest_frequency",
        full_path="rhythm.density.rest_frequency",
        description="Frequency of rests in rhythm (0.0=no rests, 1.0=many rests)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.3,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_density_sustained_note_prob",
        full_path="rhythm.density.sustained_note_prob",
        description="Probability of sustained notes (long durations)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["ballad", "ambient", "drone"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_complexity_rhythmic_diversity",
        full_path="rhythm.complexity.rhythmic_diversity",
        description="Diversity of rhythmic values used (0.0=uniform, 1.0=highly varied)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.6,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_complexity_subdivision_complexity",
        full_path="rhythm.complexity.subdivision_complexity",
        description="Complexity of rhythmic subdivisions (1=simple, 7=complex)",
        param_type=ParameterType.INTEGER,
        default_value=3,
        min_value=1,
        max_value=7,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_complexity_tuplet_usage",
        full_path="rhythm.complexity.tuplet_usage",
        description="Frequency of tuplets (triplets, quintuplets, etc., 0.0-1.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.1,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_complexity_nested_tuplet_prob",
        full_path="rhythm.complexity.nested_tuplet_prob",
        description="Probability of nested tuplets (tuplets within tuplets)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary", "experimental"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_fills_fill_probability",
        full_path="rhythm.fills.fill_probability",
        description="Probability of fills at phrase endings",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_fills_fill_density",
        full_path="rhythm.fills.fill_density",
        description="Density of notes in fills (0.0=sparse, 1.0=dense)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.7,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_fills_fill_complexity",
        full_path="rhythm.fills.fill_complexity",
        description="Complexity of fill patterns (1=simple, 7=virtuosic)",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=1,
        max_value=7,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_fills_pickup_note_prob",
        full_path="rhythm.fills.pickup_note_prob",
        description="Probability of pickup notes before phrases",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["all"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_fills_turnaround_prob",
        full_path="rhythm.fills.turnaround_prob",
        description="Probability of turnaround patterns at section endings",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "pop"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_density_silence_before_climax",
        full_path="rhythm.density.silence_before_climax",
        description="Probability of dramatic silence before climactic moments",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["dramatic", "film_score", "contemporary"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="rhythm_complexity_metric_displacement",
        full_path="rhythm.complexity.metric_displacement",
        description="Probability of metric displacement (pattern shifted in time)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "contemporary", "progressive"]
    ))


def register_all_melody_rhythm_expansion():
    """
    Register all 120 new melody and rhythm parameters

    Breakdown:
    - Melody: 63 parameters (37 → 100)
    - Rhythm: 57 parameters (43 → 100)

    Total: 120 new parameters
    """
    print("Registering Melody & Rhythm Expansion Parameters...")
    print("=" * 70)

    # Register melody parameters
    print("\n[1/2] Registering 63 Melody Parameters...")
    print("  - Melodic Contour & Shape: 20 parameters")
    print("  - Interval & Chromaticism: 23 parameters")
    print("  - Ornamentation & Articulation: 20 parameters")
    register_melody_expansion_parameters()
    print("  ✓ Melody expansion complete (37 → 100 parameters)")

    # Register rhythm parameters
    print("\n[2/2] Registering 57 Rhythm Parameters...")
    print("  - Advanced Rhythmic Patterns: 20 parameters")
    print("  - Syncopation & Feel: 22 parameters")
    print("  - Density, Complexity & Fills: 15 parameters")
    register_rhythm_expansion_parameters()
    print("  ✓ Rhythm expansion complete (43 → 100 parameters)")

    print("\n" + "=" * 70)
    print("Agent 4 Complete: 120 new parameters registered")
    print("Total new parameters: 63 (melody) + 57 (rhythm) = 120")
    print("=" * 70)


# Auto-register when module is imported
if __name__ != "__main__":
    register_all_melody_rhythm_expansion()


# ============================================================================
# PARAMETER SUMMARY & DOCUMENTATION
# ============================================================================

"""
MELODY EXPANSION SUMMARY (63 parameters)
========================================

1. MELODIC CONTOUR & SHAPE (20 parameters):
   - Contour probabilities: arch, inverted arch, ascending, descending, wave
   - Terraced dynamics, climax placement, apex emphasis
   - Range exploration, register shifts, octave displacement
   - Phrase peak placement, curve smoothness, directional consistency
   - Gap-fill principle, registral direction, tessitura
   - Peak/valley frequency, contour complexity

2. INTERVAL & CHROMATICISM (23 parameters):
   - Stepwise motion vs. leap probability
   - Maximum leap interval, leap resolution
   - Chromatic approaches: approach, passing, neighbor, enclosure, double
   - Bebop chromatic patterns
   - Diatonic passing and neighbor tones
   - Special tones: anticipation, escape, appoggiatura, cambiata
   - Tritone, augmented, diminished, compound intervals
   - Parallel intervals, consonance preference
   - Tension-resolution cycles

3. ORNAMENTATION & ARTICULATION (20 parameters):
   - Classical ornaments: trill, mordent, turn, grace notes
   - Jazz articulations: glissando, portamento, scoop, fall-off, doit
   - Expression: vibrato, tremolo, shake, bend
   - Articulations: legato, staccato, tenuto, marcato, accent
   - Slur length

RHYTHM EXPANSION SUMMARY (57 parameters)
========================================

1. ADVANCED RHYTHMIC PATTERNS (20 parameters):
   - Polyrhythm: active, ratio, cross-rhythms
   - Hemiola, metric modulation, tempo relationships
   - Polymetric layers
   - Cultural patterns: clave (son, rumba, bossa), timeline (bembe, soukous)
   - Odd meter groupings, additive vs. divisive rhythm
   - Isorhythm: talea, color lengths
   - Transformations: diminution, augmentation, retrograde, inversion, canon

2. SYNCOPATION & FEEL (22 parameters):
   - Anticipation: 8th, 16th notes
   - Delayed attack, offbeat emphasis
   - Backbeat strength, upbeat emphasis, cross-beat accents
   - Composite rhythm complexity
   - Swing ratio, shuffle intensity, straight 8ths
   - Half-time feel, double-time feel
   - Timing: laid-back, rushing, rubato
   - Agogic accent, micro-timing variation
   - Human imperfection, groove quantization
   - Pocket depth, subdivision clarity

3. DENSITY, COMPLEXITY & FILLS (15 parameters):
   - Note density, rhythmic activity level
   - Rest frequency, sustained note probability
   - Rhythmic diversity, subdivision complexity
   - Tuplet usage, nested tuplets
   - Fill probability, density, complexity
   - Pickup notes, turnarounds
   - Silence before climax, metric displacement

TOTAL: 120 NEW PARAMETERS
- Melody: 63 parameters (expanding from 37 to 100)
- Rhythm: 57 parameters (expanding from 43 to 100)

All parameters are designed for:
1. XGBoost feature extraction and prediction
2. Inverse procedural modeling
3. Genre-specific configuration
4. Musical expressiveness and authenticity
5. Backward compatibility with existing system
"""
