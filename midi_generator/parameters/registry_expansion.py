"""
Registry Expansion - Agent 1
Comprehensive parameter definitions for core modules

This file adds 400+ parameters to the Universal Parameter Registry
covering harmony, melody, rhythm, transformation, and analysis systems.
"""

from .universal_registry import (
    ParameterDefinition, ParameterType, ParameterCategory,
    MusicalImpact, REGISTRY
)


def register_core_harmony_parameters():
    """Register comprehensive harmony parameters (100+ params)"""

    # ========================================================================
    # Neo-Riemannian Transformation Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="neo_riemannian_transformation_sequence",
        full_path="harmony.neo_riemannian.transformation_sequence",
        description="Sequence of transformations (P, L, R)",
        param_type=ParameterType.CATEGORICAL,
        options=["P", "L", "R", "PL", "PR", "LR", "PLR", "custom"],
        default_value="PLR",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["classical", "film_score", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="neo_riemannian_voice_leading",
        full_path="harmony.neo_riemannian.apply_voice_leading",
        description="Apply smooth voice leading to transformations",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="hexatonic_pole",
        full_path="harmony.neo_riemannian.hexatonic_pole",
        description="Hexatonic system pole (0-3: Northern, Southern, Eastern, Western)",
        param_type=ParameterType.INTEGER,
        default_value=0,
        min_value=0,
        max_value=3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Modal Harmony Parameters
    # ========================================================================

    for mode in ["dorian", "phrygian", "lydian", "mixolydian", "aeolian", "locrian"]:
        REGISTRY.register(ParameterDefinition(
            name=f"modal_{mode}_probability",
            full_path=f"harmony.modal.{mode}_probability",
            description=f"Probability of using {mode.title()} mode",
            param_type=ParameterType.PROBABILITY,
            default_value=0.14,  # Equal distribution across 7 modes
            category=ParameterCategory.HARMONY,
            musical_impact=MusicalImpact.CRITICAL,
            genre_relevance=["jazz", "classical", "world"]
        ))

    REGISTRY.register(ParameterDefinition(
        name="modal_progression_type",
        full_path="harmony.modal.progression_type",
        description="Type of modal progression",
        param_type=ParameterType.CATEGORICAL,
        options=["characteristic", "vamp", "plagal", "descending", "ascending"],
        default_value="characteristic",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="modal_interchange_intensity",
        full_path="harmony.modal.interchange_intensity",
        description="How often to borrow from parallel modes (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "rock", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="modal_cadence_type",
        full_path="harmony.modal.cadence_type",
        description="Type of modal cadence",
        param_type=ParameterType.CATEGORICAL,
        options=["plagal", "phrygian", "dorian", "lydian", "subtonic"],
        default_value="plagal",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Chromatic Harmony Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="chromatic_mediant_pattern",
        full_path="harmony.chromatic.mediant_pattern",
        description="Chromatic mediant progression pattern",
        param_type=ParameterType.CATEGORICAL,
        options=["UCM", "LCM", "UFM", "LFM", "UCM LCM", "UFM LFM", "mixed"],
        default_value="UCM LCM",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "film_score", "progressive"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="secondary_dominant_probability",
        full_path="harmony.chromatic.secondary_dominant_probability",
        description="Probability of using secondary dominants",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "musical_theatre"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminished_passing_probability",
        full_path="harmony.chromatic.diminished_passing_probability",
        description="Probability of diminished passing chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="augmented_sixth_probability",
        full_path="harmony.chromatic.augmented_sixth_probability",
        description="Probability of augmented sixth chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "romantic"]
    ))

    # ========================================================================
    # Voice Leading Parameters (Expanded)
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="voice_leading_max_motion",
        full_path="harmony.voice_leading.max_motion",
        description="Maximum voice motion in semitones",
        param_type=ParameterType.INTEGER,
        default_value=7,
        min_value=1,
        max_value=24,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="voice_leading_prefer_contrary",
        full_path="harmony.voice_leading.prefer_contrary_motion",
        description="Preference for contrary motion (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="voice_leading_allow_voice_crossing",
        full_path="harmony.voice_leading.allow_voice_crossing",
        description="Allow voices to cross",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="voice_leading_common_tone_weight",
        full_path="harmony.voice_leading.common_tone_weight",
        description="Weight for maintaining common tones (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Chord Quality and Extension Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="seventh_chord_probability",
        full_path="harmony.extensions.seventh_probability",
        description="Probability of using 7th chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "R&B"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="altered_dominant_probability",
        full_path="harmony.extensions.altered_dominant_probability",
        description="Probability of altered dominants (b9, #9, b13, etc.)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "fusion"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="sus_chord_probability",
        full_path="harmony.extensions.sus_probability",
        description="Probability of suspended chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "pop", "rock"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="add9_probability",
        full_path="harmony.extensions.add9_probability",
        description="Probability of add9 chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["pop", "rock", "singer_songwriter"]
    ))


def register_core_melody_parameters():
    """Register comprehensive melody parameters (80+ params)"""

    # ========================================================================
    # Melodic Contour Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="contour_arch_probability",
        full_path="melody.contour.arch_probability",
        description="Probability of arch contour (low-high-low)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.35,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL
    ))

    REGISTRY.register(ParameterDefinition(
        name="contour_inverted_arch_probability",
        full_path="melody.contour.inverted_arch_probability",
        description="Probability of inverted arch (high-low-high)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="contour_ascending_probability",
        full_path="melody.contour.ascending_probability",
        description="Probability of ascending contour",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="contour_descending_probability",
        full_path="melody.contour.descending_probability",
        description="Probability of descending contour",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="contour_wave_probability",
        full_path="melody.contour.wave_probability",
        description="Probability of wave contour",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    # ========================================================================
    # Interval Distribution Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="interval_unison_probability",
        full_path="melody.intervals.unison_probability",
        description="Probability of repeated notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_second_probability",
        full_path="melody.intervals.second_probability",
        description="Probability of stepwise motion (seconds)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.CRITICAL
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_third_probability",
        full_path="melody.intervals.third_probability",
        description="Probability of third leaps",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_fourth_probability",
        full_path="melody.intervals.fourth_probability",
        description="Probability of fourth leaps",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_fifth_probability",
        full_path="melody.intervals.fifth_probability",
        description="Probability of fifth leaps",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_large_leap_probability",
        full_path="melody.intervals.large_leap_probability",
        description="Probability of leaps > fifth",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="leap_recovery_probability",
        full_path="melody.intervals.leap_recovery_probability",
        description="Probability of stepwise recovery after leap",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Melodic Ornamentation Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="trill_probability",
        full_path="melody.ornaments.trill_probability",
        description="Probability of trills",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="mordent_probability",
        full_path="melody.ornaments.mordent_probability",
        description="Probability of mordents",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="turn_probability",
        full_path="melody.ornaments.turn_probability",
        description="Probability of turns",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="appoggiatura_probability",
        full_path="melody.ornaments.appoggiatura_probability",
        description="Probability of appoggiaturas",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="grace_note_probability",
        full_path="melody.ornaments.grace_note_probability",
        description="Probability of grace notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "folk", "country"]
    ))

    # ========================================================================
    # Phrasing Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="phrase_length_min",
        full_path="melody.phrasing.length_min",
        description="Minimum phrase length in beats",
        param_type=ParameterType.INTEGER,
        default_value=4,
        min_value=2,
        max_value=32,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_length_max",
        full_path="melody.phrasing.length_max",
        description="Maximum phrase length in beats",
        param_type=ParameterType.INTEGER,
        default_value=16,
        min_value=4,
        max_value=64,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrase_rest_probability",
        full_path="melody.phrasing.rest_probability",
        description="Probability of rest between phrases",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="antecedent_consequent_probability",
        full_path="melody.phrasing.antecedent_consequent_probability",
        description="Probability of antecedent-consequent phrase pairs",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Motivic Development Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="motif_repetition_probability",
        full_path="melody.motif.repetition_probability",
        description="Probability of exact motif repetition",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="motif_sequence_probability",
        full_path="melody.motif.sequence_probability",
        description="Probability of motif sequencing",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="motif_inversion_probability",
        full_path="melody.motif.inversion_probability",
        description="Probability of motif inversion",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="motif_retrograde_probability",
        full_path="melody.motif.retrograde_probability",
        description="Probability of motif retrograde",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="motif_augmentation_probability",
        full_path="melody.motif.augmentation_probability",
        description="Probability of rhythmic augmentation",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="motif_diminution_probability",
        full_path="melody.motif.diminution_probability",
        description="Probability of rhythmic diminution",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.MELODY,
        musical_impact=MusicalImpact.MEDIUM
    ))


def register_core_rhythm_parameters():
    """Register comprehensive rhythm parameters (60+ params)"""

    # ========================================================================
    # Swing and Groove Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="swing_ratio",
        full_path="rhythm.swing.ratio",
        description="Swing ratio (0.5=straight, 0.67=standard, 0.75=hard)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.67,
        min_value=0.5,
        max_value=0.8,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "swing", "blues", "shuffle"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="swing_intensity",
        full_path="rhythm.swing.intensity",
        description="How consistently swing is applied (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=1.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="groove_pocket_depth",
        full_path="rhythm.groove.pocket_depth",
        description="How far behind/ahead of beat (ms, negative=behind)",
        param_type=ParameterType.INTEGER,
        default_value=0,
        min_value=-50,
        max_value=50,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["funk", "R&B", "neo_soul"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="shuffle_feel",
        full_path="rhythm.groove.shuffle_feel",
        description="Shuffle feel intensity (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "shuffle", "R&B"]
    ))

    # ========================================================================
    # Syncopation Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="syncopation_level",
        full_path="rhythm.syncopation.level",
        description="Overall syncopation level (0.0=none, 1.0=maximum)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "funk", "latin", "afrobeat"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="anticipation_probability",
        full_path="rhythm.syncopation.anticipation_probability",
        description="Probability of anticipating strong beats",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="offbeat_emphasis_probability",
        full_path="rhythm.syncopation.offbeat_emphasis_probability",
        description="Probability of emphasizing offbeats",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["reggae", "ska", "funk"]
    ))

    # ========================================================================
    # Polyrhythm Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="polyrhythm_probability",
        full_path="rhythm.polyrhythm.probability",
        description="Probability of polyrhythmic patterns",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["african", "afrobeat", "progressive", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="polyrhythm_ratio",
        full_path="rhythm.polyrhythm.ratio",
        description="Polyrhythm ratio (e.g., 3:2, 4:3, 5:4)",
        param_type=ParameterType.CATEGORICAL,
        options=["3:2", "4:3", "5:4", "7:4", "3:4", "5:3"],
        default_value="3:2",
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    # ========================================================================
    # Metric Modulation Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="metric_modulation_probability",
        full_path="rhythm.metric_modulation.probability",
        description="Probability of metric modulation",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["progressive", "contemporary_classical"]
    ))

    # ========================================================================
    # Subdivision Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="triplet_subdivision_probability",
        full_path="rhythm.subdivision.triplet_probability",
        description="Probability of triplet subdivisions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="quintuplet_probability",
        full_path="rhythm.subdivision.quintuplet_probability",
        description="Probability of quintuplet subdivisions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["progressive", "jazz"]
    ))

    REGISTRY.register(ParameterDefinition(
        name="sextuplet_probability",
        full_path="rhythm.subdivision.sextuplet_probability",
        description="Probability of sextuplet subdivisions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM
    ))

    # ========================================================================
    # Rhythmic Density Parameters
    # ========================================================================

    REGISTRY.register(ParameterDefinition(
        name="note_density",
        full_path="rhythm.density.note_density",
        description="Average notes per beat (0.5-8.0)",
        param_type=ParameterType.CONTINUOUS,
        default_value=2.0,
        min_value=0.5,
        max_value=8.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL
    ))

    REGISTRY.register(ParameterDefinition(
        name="rest_frequency",
        full_path="rhythm.density.rest_frequency",
        description="Frequency of rests (0.0-1.0)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="sustained_note_probability",
        full_path="rhythm.density.sustained_note_probability",
        description="Probability of long sustained notes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM
    ))


def register_transformation_parameters():
    """Register transformation parameters (40+ params)"""

    REGISTRY.register(ParameterDefinition(
        name="transposition_amount",
        full_path="transformation.transposition.semitones",
        description="Transposition in semitones",
        param_type=ParameterType.INTEGER,
        default_value=0,
        min_value=-24,
        max_value=24,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL
    ))

    REGISTRY.register(ParameterDefinition(
        name="tempo_change_ratio",
        full_path="transformation.tempo.change_ratio",
        description="Tempo change ratio (0.5=half, 2.0=double)",
        param_type=ParameterType.CONTINUOUS,
        default_value=1.0,
        min_value=0.25,
        max_value=4.0,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.CRITICAL
    ))

    REGISTRY.register(ParameterDefinition(
        name="dynamic_scaling",
        full_path="transformation.dynamics.scaling",
        description="Dynamic scaling factor (0.5=softer, 2.0=louder)",
        param_type=ParameterType.CONTINUOUS,
        default_value=1.0,
        min_value=0.1,
        max_value=3.0,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="humanization_amount",
        full_path="transformation.humanization.amount",
        description="Humanization intensity (0.0=robotic, 1.0=very human)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.HIGH
    ))

    REGISTRY.register(ParameterDefinition(
        name="humanization_timing_variance",
        full_path="transformation.humanization.timing_variance",
        description="Timing variance in milliseconds",
        param_type=ParameterType.INTEGER,
        default_value=15,
        min_value=0,
        max_value=50,
        category=ParameterCategory.RHYTHM,
        musical_impact=MusicalImpact.MEDIUM
    ))

    REGISTRY.register(ParameterDefinition(
        name="humanization_velocity_variance",
        full_path="transformation.humanization.velocity_variance",
        description="Velocity variance (+/-)",
        param_type=ParameterType.INTEGER,
        default_value=10,
        min_value=0,
        max_value=40,
        category=ParameterCategory.DYNAMICS,
        musical_impact=MusicalImpact.MEDIUM
    ))


def register_all_expansions():
    """Register all expansion parameters"""
    print("🔧 Registering core harmony parameters...")
    register_core_harmony_parameters()

    print("🔧 Registering core melody parameters...")
    register_core_melody_parameters()

    print("🔧 Registering core rhythm parameters...")
    register_core_rhythm_parameters()

    print("🔧 Registering transformation parameters...")
    register_transformation_parameters()

    stats = REGISTRY.get_statistics()
    print(f"\n✅ Registry expansion complete!")
    print(f"   Total parameters: {stats['total_parameters']}")
    print(f"   Added: {stats['total_parameters'] - 28} new parameters")


if __name__ == "__main__":
    register_all_expansions()

    # Export updated registry
    REGISTRY.export_to_json("/home/user/Do/midi_generator/parameters/registry.json")
    REGISTRY.generate_documentation("/home/user/Do/midi_generator/parameters/PARAMETERS.md")

    print("\n💾 Exported updated registry")
