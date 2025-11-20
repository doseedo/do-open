"""
Harmony Deep Expansion Module - Agent 3
=======================================

Expands harmony parameters from 56 to 150 total parameters (+94 new)

This module implements advanced harmony theory concepts including:
- Advanced Voicing Techniques (25 parameters)
- Advanced Chord Extensions (20 parameters)
- Advanced Progressions (25 parameters)
- Neo-Riemannian & Advanced Theory (14 parameters)
- Specialized Techniques (10 parameters)

Musical Theory Foundation:
--------------------------
This expansion covers professional-level harmony concepts used in:
- Jazz (bebop, modal, free jazz)
- Classical (romantic, contemporary)
- Film scoring and game music
- Progressive rock and fusion
- World music traditions

Author: Agent 3 - Harmony Deep Expansion Specialist
License: MIT
Version: 1.0.0
"""

from .universal_registry import (
    ParameterDefinition, ParameterType, ParameterCategory,
    MusicalImpact, REGISTRY
)


# ============================================================================
# SECTION 1: ADVANCED VOICING TECHNIQUES (25 parameters)
# ============================================================================

def register_advanced_voicing_techniques():
    """
    Register 25 advanced voicing technique parameters.

    Voicing refers to how chord tones are arranged vertically and horizontally.
    Advanced voicings include quartal/quintal harmony, clusters, polychords,
    drop voicings, and sophisticated jazz techniques.
    """

    # ------------------------------------------------------------------------
    # Quartal and Quintal Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="quartal_probability",
        full_path="harmony.voicing.quartal_probability",
        description="Probability of using quartal voicings (stacked fourths)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "modal_jazz", "fusion", "contemporary_classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Used heavily in McCoy Tyner style, modal jazz, and modern film scores"
    ))

    REGISTRY.register(ParameterDefinition(
        name="quintal_probability",
        full_path="harmony.voicing.quintal_probability",
        description="Probability of using quintal voicings (stacked fifths)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "ambient", "new_age"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates open, medieval or modern atmospheric sound"
    ))

    # ------------------------------------------------------------------------
    # Cluster Voicings
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="cluster_probability",
        full_path="harmony.voicing.cluster_probability",
        description="Probability of using tone clusters (closely spaced dissonances)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary_classical", "avant_garde", "film_score", "horror"],
        module_file="generators/harmony_module.py",
        constraint_description="Used by composers like Henry Cowell, Ligeti. Creates dense, dissonant textures"
    ))

    REGISTRY.register(ParameterDefinition(
        name="cluster_width",
        full_path="harmony.voicing.cluster_width",
        description="Width of tone clusters in semitones",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 3, 4, 5],
        default_value=3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "avant_garde"],
        module_file="generators/harmony_module.py",
        constraint_description="2=minor 2nd cluster, 3=minor 3rd, 4=major 3rd, 5=perfect 4th"
    ))

    # ------------------------------------------------------------------------
    # Upper Structure and Polychords
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="upper_structure_triad",
        full_path="harmony.voicing.upper_structure_triad",
        description="Probability of adding upper structure triads over bass",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "fusion", "gospel"],
        module_file="generators/harmony_module.py",
        constraint_description="E.g., D/C7 creates C7#9#11. Common in Barry Harris method"
    ))

    REGISTRY.register(ParameterDefinition(
        name="polychord_probability",
        full_path="harmony.voicing.polychord_probability",
        description="Probability of using polychords (two simultaneous triads)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "contemporary_classical", "progressive_rock"],
        module_file="generators/harmony_module.py",
        constraint_description="E.g., Eb/C creates C7#9. Used by Stravinsky, Bartok, Messiaen"
    ))

    REGISTRY.register(ParameterDefinition(
        name="bitonal_probability",
        full_path="harmony.voicing.bitonal_probability",
        description="Probability of true bitonality (two independent key centers)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.02,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary_classical", "avant_garde", "experimental"],
        module_file="generators/harmony_module.py",
        constraint_description="Extreme dissonance. Used sparingly even by modernists"
    ))

    # ------------------------------------------------------------------------
    # Jazz Voicing Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="shell_voicing_prob",
        full_path="harmony.voicing.shell_voicing_prob",
        description="Probability of using shell voicings (root, 3rd, 7th only)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bossa_nova", "latin_jazz"],
        module_file="generators/harmony_module.py",
        constraint_description="Minimal voicing emphasizing chord quality. Freddie Green style"
    ))

    REGISTRY.register(ParameterDefinition(
        name="rootless_prob",
        full_path="harmony.voicing.rootless_prob",
        description="Probability of rootless voicings (3rd, 7th, extensions, no root)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Essential jazz piano technique. Root supplied by bass"
    ))

    REGISTRY.register(ParameterDefinition(
        name="so_what_voicing",
        full_path="harmony.voicing.so_what_voicing",
        description="Probability of 'So What' voicings (3 perfect 4ths + major 3rd)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["modal_jazz", "jazz", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Bill Evans voicing from Miles Davis 'So What'. Iconic modal sound"
    ))

    # ------------------------------------------------------------------------
    # Drop Voicings
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="drop2_prob",
        full_path="harmony.voicing.drop2_prob",
        description="Probability of drop-2 voicings (2nd voice from top dropped octave)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Most common orchestral/jazz voicing. Creates balanced spread"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drop3_prob",
        full_path="harmony.voicing.drop3_prob",
        description="Probability of drop-3 voicings (3rd voice from top dropped octave)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Wider spread than drop-2. Common in horn sections"
    ))

    REGISTRY.register(ParameterDefinition(
        name="drop24_prob",
        full_path="harmony.voicing.drop24_prob",
        description="Probability of drop-2-and-4 voicings (both 2nd and 4th dropped)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "big_band"],
        module_file="generators/harmony_module.py",
        constraint_description="Very wide spread. Used in guitar and large ensemble writing"
    ))

    # ------------------------------------------------------------------------
    # Voicing Density and Spread
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="spread_voicing_prob",
        full_path="harmony.voicing.spread_voicing_prob",
        description="Probability of spread voicings (wide intervals between voices)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "orchestral", "film_score"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates transparent, spacious texture"
    ))

    REGISTRY.register(ParameterDefinition(
        name="close_position_prob",
        full_path="harmony.voicing.close_position_prob",
        description="Probability of close position voicings (all voices within octave)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral", "hymn"],
        module_file="generators/harmony_module.py",
        constraint_description="Traditional SATB writing. Compact, blended sound"
    ))

    REGISTRY.register(ParameterDefinition(
        name="open_position_prob",
        full_path="harmony.voicing.open_position_prob",
        description="Probability of open position voicings (voices span > 1 octave)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral"],
        module_file="generators/harmony_module.py",
        constraint_description="More resonant than close position. Classical standard"
    ))

    # ------------------------------------------------------------------------
    # Voice Doubling Options
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="doubled_root",
        full_path="harmony.voicing.doubled_root",
        description="Whether to double the root in voicings",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral", "hymn"],
        module_file="generators/harmony_module.py",
        constraint_description="Most common doubling. Reinforces tonal center"
    ))

    REGISTRY.register(ParameterDefinition(
        name="doubled_third",
        full_path="harmony.voicing.doubled_third",
        description="Whether to double the third in voicings",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "choral"],
        module_file="generators/harmony_module.py",
        constraint_description="Less common. Can create interesting color in major chords"
    ))

    REGISTRY.register(ParameterDefinition(
        name="doubled_fifth",
        full_path="harmony.voicing.doubled_fifth",
        description="Whether to double the fifth in voicings",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.LOW,
        genre_relevance=["classical", "choral"],
        module_file="generators/harmony_module.py",
        constraint_description="Common in classical harmony. Neutral doubling"
    ))

    REGISTRY.register(ParameterDefinition(
        name="doubled_seventh",
        full_path="harmony.voicing.doubled_seventh",
        description="Whether to double the seventh in voicings",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "contemporary"],
        module_file="generators/harmony_module.py",
        constraint_description="Unusual. Can create pungent dissonance"
    ))

    # ------------------------------------------------------------------------
    # Note Omission Options
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="omit_root",
        full_path="harmony.voicing.omit_root",
        description="Whether to omit root (assumes bass provides it)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "latin", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Standard in jazz comping. Makes room for extensions"
    ))

    REGISTRY.register(ParameterDefinition(
        name="omit_fifth",
        full_path="harmony.voicing.omit_fifth",
        description="Whether to omit fifth (least important chord tone)",
        param_type=ParameterType.BOOLEAN,
        default_value=True,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "pop", "R&B"],
        module_file="generators/harmony_module.py",
        constraint_description="Very common. Fifth is often expendable in modern harmony"
    ))

    # ------------------------------------------------------------------------
    # Tone Emphasis and Color
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="guide_tone_emphasis",
        full_path="harmony.voicing.guide_tone_emphasis",
        description="How much to emphasize guide tones (3rd and 7th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.8,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "latin"],
        module_file="generators/harmony_module.py",
        constraint_description="Guide tones define chord quality and create smooth voice leading"
    ))

    REGISTRY.register(ParameterDefinition(
        name="color_tone_emphasis",
        full_path="harmony.voicing.color_tone_emphasis",
        description="How much to emphasize color tones (9th, 11th, 13th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "fusion", "neo_soul"],
        module_file="generators/harmony_module.py",
        constraint_description="Extensions add sophistication and harmonic richness"
    ))

    # ------------------------------------------------------------------------
    # Avoid Note Handling
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="avoid_note_handling",
        full_path="harmony.voicing.avoid_note_handling",
        description="How to handle avoid notes (e.g., 4th over major chord)",
        param_type=ParameterType.CATEGORICAL,
        options=["include", "avoid", "approach_from_below", "approach_from_above"],
        default_value="avoid",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Avoid notes clash with chord. Traditional jazz theory excludes them"
    ))


# ============================================================================
# SECTION 2: ADVANCED CHORD EXTENSIONS (20 parameters)
# ============================================================================

def register_advanced_chord_extensions():
    """
    Register 20 advanced chord extension parameters.

    Extensions are notes added beyond the 7th: 9ths, 11ths, 13ths, and their
    alterations. These create harmonic sophistication and color.
    """

    # ------------------------------------------------------------------------
    # Ninth Extensions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="ninth_usage",
        full_path="harmony.extension.ninth_usage",
        description="Overall probability of including 9th extensions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.7,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "R&B", "neo_soul", "gospel"],
        module_file="generators/harmony_module.py",
        constraint_description="9ths are most consonant extension, used extensively in modern music"
    ))

    REGISTRY.register(ParameterDefinition(
        name="flat_ninth_prob",
        full_path="harmony.extension.flat_ninth_prob",
        description="Probability of b9 (typically on dominant chords)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "flamenco"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates tension and Spanish/Phrygian flavor. 7b9 is classic bebop sound"
    ))

    REGISTRY.register(ParameterDefinition(
        name="sharp_ninth_prob",
        full_path="harmony.extension.sharp_ninth_prob",
        description="Probability of #9 (Hendrix chord when on dominant)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "rock", "funk"],
        module_file="generators/harmony_module.py",
        constraint_description="7#9 is the 'Hendrix chord'. Bluesy, funky sound"
    ))

    # ------------------------------------------------------------------------
    # Eleventh Extensions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="eleventh_usage",
        full_path="harmony.extension.eleventh_usage",
        description="Overall probability of including 11th extensions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "fusion", "modal"],
        module_file="generators/harmony_module.py",
        constraint_description="11th requires care - avoid note on major chords. Safe on minor, dominant"
    ))

    REGISTRY.register(ParameterDefinition(
        name="sharp_eleventh_prob",
        full_path="harmony.extension.sharp_eleventh_prob",
        description="Probability of #11 (Lydian sound)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "lydian", "film_score"],
        module_file="generators/harmony_module.py",
        constraint_description="#11 is Lydian characteristic tone. Bright, modern, cinematic"
    ))

    # ------------------------------------------------------------------------
    # Thirteenth Extensions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="thirteenth_usage",
        full_path="harmony.extension.thirteenth_usage",
        description="Overall probability of including 13th extensions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "big_band", "swing"],
        module_file="generators/harmony_module.py",
        constraint_description="13th chords are full, lush. Common in swing, big band, R&B"
    ))

    REGISTRY.register(ParameterDefinition(
        name="flat_thirteenth_prob",
        full_path="harmony.extension.flat_thirteenth_prob",
        description="Probability of b13 (enharmonic with #5)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "altered", "bebop"],
        module_file="generators/harmony_module.py",
        constraint_description="b13 part of altered scale. Creates exotic tension"
    ))

    # ------------------------------------------------------------------------
    # Add Tones (non-tertian extensions)
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="add2_prob",
        full_path="harmony.extension.add2_prob",
        description="Probability of add2 chords (1-2-3-5, no 7th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["pop", "rock", "indie", "singer_songwriter"],
        module_file="generators/harmony_module.py",
        constraint_description="Bright, open sound. Popular in acoustic pop, rock ballads"
    ))

    REGISTRY.register(ParameterDefinition(
        name="add4_prob",
        full_path="harmony.extension.add4_prob",
        description="Probability of add4 chords (1-3-4-5, creates dissonance)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["pop", "rock"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates tension, less common than add2 or add9"
    ))

    REGISTRY.register(ParameterDefinition(
        name="add6_prob",
        full_path="harmony.extension.add6_prob",
        description="Probability of add6 chords (major 6th, no 7th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "swing", "bossa_nova", "french_pop"],
        module_file="generators/harmony_module.py",
        constraint_description="Sweet, nostalgic sound. Common ending chord in jazz standards"
    ))

    # ------------------------------------------------------------------------
    # Suspended Chords
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="sus2_prob",
        full_path="harmony.extension.sus2_prob",
        description="Probability of sus2 chords (1-2-5, no 3rd)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "ambient", "new_age"],
        module_file="generators/harmony_module.py",
        constraint_description="Open, ambiguous. Neither major nor minor. Creates floating quality"
    ))

    REGISTRY.register(ParameterDefinition(
        name="sus4_prob",
        full_path="harmony.extension.sus4_prob",
        description="Probability of sus4 chords (1-4-5, no 3rd)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "folk", "CCM"],
        module_file="generators/harmony_module.py",
        constraint_description="Suspended resolution. Traditional tension-release mechanism"
    ))

    # ------------------------------------------------------------------------
    # Altered Qualities
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="augmented_prob",
        full_path="harmony.extension.augmented_prob",
        description="Probability of augmented chords (raised 5th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "impressionist"],
        module_file="generators/harmony_module.py",
        constraint_description="Whole-tone sound. Used in impressionism, jazz. Creates instability"
    ))

    REGISTRY.register(ParameterDefinition(
        name="diminished_prob",
        full_path="harmony.extension.diminished_prob",
        description="Probability of fully diminished 7th chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical", "tango"],
        module_file="generators/harmony_module.py",
        constraint_description="Symmetrical structure. Versatile passing chord. Very tense"
    ))

    REGISTRY.register(ParameterDefinition(
        name="half_diminished_prob",
        full_path="harmony.extension.half_diminished_prob",
        description="Probability of half-diminished 7th chords (m7b5)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bossa_nova", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="ii chord in minor. Essential in ii-V-i progressions"
    ))

    # ------------------------------------------------------------------------
    # Dominant Alterations
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="altered_dominant_prob",
        full_path="harmony.extension.altered_dominant_prob",
        description="Probability of fully altered dominants (b9,#9,b5,b13)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "bebop", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Altered scale. Maximum tension on dominant. Bebop essential"
    ))

    REGISTRY.register(ParameterDefinition(
        name="lydian_dominant_prob",
        full_path="harmony.extension.lydian_dominant_prob",
        description="Probability of Lydian dominant (7#11)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "fusion", "modal"],
        module_file="generators/harmony_module.py",
        constraint_description="4th mode of melodic minor. Bright, modern dominant sound"
    ))

    REGISTRY.register(ParameterDefinition(
        name="phrygian_dominant_prob",
        full_path="harmony.extension.phrygian_dominant_prob",
        description="Probability of Phrygian dominant (7b9b13)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["flamenco", "spanish", "klezmer", "middle_eastern"],
        module_file="generators/harmony_module.py",
        constraint_description="5th mode of harmonic minor. Exotic, Spanish flavor"
    ))

    REGISTRY.register(ParameterDefinition(
        name="dominant_bebop_prob",
        full_path="harmony.extension.dominant_bebop_prob",
        description="Probability of bebop dominant scale usage",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["bebop", "jazz", "swing"],
        module_file="generators/harmony_module.py",
        constraint_description="Mixolydian + major 7th. Creates chromatic passing tones"
    ))

    # ------------------------------------------------------------------------
    # Interval Stacking Methods
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="stacked_intervals",
        full_path="harmony.extension.stacked_intervals",
        description="Primary interval for chord construction",
        param_type=ParameterType.CATEGORICAL,
        options=["thirds", "fourths", "fifths", "mixed"],
        default_value="thirds",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        module_file="generators/harmony_module.py",
        constraint_description="Thirds=traditional, fourths=quartal, fifths=quintal, mixed=eclectic"
    ))


# ============================================================================
# SECTION 3: ADVANCED PROGRESSIONS (25 parameters)
# ============================================================================

def register_advanced_progressions():
    """
    Register 25 advanced progression parameters.

    Progressions define how chords move through time. Advanced progressions
    include substitutions, reharmonization, modal interchange, and complex
    cadential patterns.
    """

    # ------------------------------------------------------------------------
    # Substitution Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="tritone_substitution",
        full_path="harmony.progression.tritone_substitution",
        description="Probability of tritone substitution (substitute dominant)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "bebop", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Replace V7 with bII7. Shares tritone. Creates chromatic bass motion"
    ))

    REGISTRY.register(ParameterDefinition(
        name="chromatic_mediant",
        full_path="harmony.progression.chromatic_mediant",
        description="Probability of chromatic mediant relationships",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "film_score", "progressive_rock"],
        module_file="generators/harmony_module.py",
        constraint_description="Chords whose roots are a third apart. Wagner, Brahms, film music"
    ))

    REGISTRY.register(ParameterDefinition(
        name="modal_interchange",
        full_path="harmony.progression.modal_interchange",
        description="Probability of borrowing chords from parallel modes",
        param_type=ParameterType.PROBABILITY,
        default_value=0.4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "rock", "pop", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Borrow from parallel major/minor. E.g., bVI, bVII in major"
    ))

    # ------------------------------------------------------------------------
    # Secondary Functions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="secondary_dominant_prob",
        full_path="harmony.progression.secondary_dominant_prob",
        description="Probability of secondary dominants (V7/x)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "jazz", "musical_theatre"],
        module_file="generators/harmony_module.py",
        constraint_description="Dominant of a scale degree other than I. Creates forward motion"
    ))

    REGISTRY.register(ParameterDefinition(
        name="secondary_diminished_prob",
        full_path="harmony.progression.secondary_diminished_prob",
        description="Probability of secondary diminished chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Diminished approach to any chord. Common in jazz and classical"
    ))

    # ------------------------------------------------------------------------
    # Cadence Types
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="deceptive_cadence_prob",
        full_path="harmony.progression.deceptive_cadence_prob",
        description="Probability of deceptive cadences (V-vi instead of V-I)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "pop", "jazz"],
        module_file="generators/harmony_module.py",
        constraint_description="Delays resolution. Creates surprise and extends phrases"
    ))

    REGISTRY.register(ParameterDefinition(
        name="plagal_cadence_prob",
        full_path="harmony.progression.plagal_cadence_prob",
        description="Probability of plagal cadences (IV-I, 'Amen' cadence)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["hymn", "gospel", "rock", "pop"],
        module_file="generators/harmony_module.py",
        constraint_description="Softer than authentic cadence. Common in hymns, rock anthems"
    ))

    REGISTRY.register(ParameterDefinition(
        name="half_cadence_prob",
        full_path="harmony.progression.half_cadence_prob",
        description="Probability of half cadences (ending on V)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates suspense. Demands continuation"
    ))

    # ------------------------------------------------------------------------
    # Jazz-Specific Progressions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="turnaround_type",
        full_path="harmony.progression.turnaround_type",
        description="Type of turnaround progression",
        param_type=ParameterType.CATEGORICAL,
        options=["I-VI-II-V", "I-III-VI-II-V", "rhythm_changes", "bird_blues", "custom"],
        default_value="I-VI-II-V",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "swing", "bebop"],
        module_file="generators/harmony_module.py",
        constraint_description="Progression that cycles back to I. Essential jazz device"
    ))

    REGISTRY.register(ParameterDefinition(
        name="cycle_movement",
        full_path="harmony.progression.cycle_movement",
        description="Type of cycle movement for progressions",
        param_type=ParameterType.CATEGORICAL,
        options=["fifths", "fourths", "thirds", "chromatic"],
        default_value="fifths",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Circle of fifths most common. Fourths ascending = fifths descending"
    ))

    REGISTRY.register(ParameterDefinition(
        name="coltrane_changes",
        full_path="harmony.progression.coltrane_changes",
        description="Use Coltrane changes (major thirds cycle)",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "avant_garde", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Tonal centers a major third apart. From 'Giant Steps'"
    ))

    REGISTRY.register(ParameterDefinition(
        name="giant_steps_prob",
        full_path="harmony.progression.giant_steps_prob",
        description="Probability of Giant Steps substitution patterns",
        param_type=ParameterType.PROBABILITY,
        default_value=0.05,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Extreme harmonic complexity. Advanced jazz concept"
    ))

    REGISTRY.register(ParameterDefinition(
        name="backdoor_progression",
        full_path="harmony.progression.backdoor_progression",
        description="Probability of backdoor progression (bVII7-I)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "blues", "rock"],
        module_file="generators/harmony_module.py",
        constraint_description="Alternative to V7-I. Smoother, less tension"
    ))

    # ------------------------------------------------------------------------
    # Static and Pedal Point Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="pedal_point_harmony",
        full_path="harmony.progression.pedal_point_harmony",
        description="Probability of using pedal point (sustained bass note)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "organ", "ambient", "drone"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates tension and release. Common in baroque, contemporary"
    ))

    REGISTRY.register(ParameterDefinition(
        name="static_harmony_duration",
        full_path="harmony.progression.static_harmony_duration",
        description="Duration of static harmony in bars",
        param_type=ParameterType.CATEGORICAL,
        options=[2, 4, 8, 16],
        default_value=4,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["modal", "ambient", "minimalist"],
        module_file="generators/harmony_module.py",
        constraint_description="How long to stay on one chord. Modal music uses extended stasis"
    ))

    # ------------------------------------------------------------------------
    # Harmonic Rhythm
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="harmonic_rhythm",
        full_path="harmony.progression.harmonic_rhythm",
        description="Overall harmonic rhythm (rate of chord change)",
        param_type=ParameterType.CATEGORICAL,
        options=["slow", "medium", "fast", "variable"],
        default_value="medium",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        module_file="generators/harmony_module.py",
        constraint_description="Slow=ballad, medium=standard, fast=bebop, variable=dynamic"
    ))

    REGISTRY.register(ParameterDefinition(
        name="chords_per_bar",
        full_path="harmony.progression.chords_per_bar",
        description="Average number of chords per bar",
        param_type=ParameterType.CONTINUOUS,
        default_value=1.0,
        min_value=0.25,
        max_value=4.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["all"],
        module_file="generators/harmony_module.py",
        constraint_description="0.25=one chord per 4 bars, 4.0=4 chords per bar (bebop)"
    ))

    # ------------------------------------------------------------------------
    # Reharmonization Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="reharmonization_prob",
        full_path="harmony.progression.reharmonization_prob",
        description="Probability of reharmonizing existing progressions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "bebop", "fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Replace simple chords with complex substitutions"
    ))

    REGISTRY.register(ParameterDefinition(
        name="polytonality",
        full_path="harmony.progression.polytonality",
        description="Degree of polytonal writing (multiple keys simultaneously)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary_classical", "avant_garde"],
        module_file="generators/harmony_module.py",
        constraint_description="Advanced 20th century technique. Stravinsky, Bartok, Milhaud"
    ))

    # ------------------------------------------------------------------------
    # Parallel Motion Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="parallel_harmony",
        full_path="harmony.progression.parallel_harmony",
        description="Probability of parallel chord motion (planing)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["impressionist", "jazz", "film_score"],
        module_file="generators/harmony_module.py",
        constraint_description="Moving chord shapes in parallel. Debussy's signature technique"
    ))

    REGISTRY.register(ParameterDefinition(
        name="planing_type",
        full_path="harmony.progression.planing_type",
        description="Type of parallel motion",
        param_type=ParameterType.CATEGORICAL,
        options=["none", "diatonic", "chromatic", "quartal"],
        default_value="none",
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["impressionist", "jazz"],
        module_file="generators/harmony_module.py",
        constraint_description="Diatonic=scale-based, chromatic=exact intervals, quartal=4ths"
    ))

    REGISTRY.register(ParameterDefinition(
        name="upper_structure_reharmonization",
        full_path="harmony.progression.upper_structure_reharmonization",
        description="Probability of upper structure reharmonization",
        param_type=ParameterType.PROBABILITY,
        default_value=0.25,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "gospel", "neo_soul"],
        module_file="generators/harmony_module.py",
        constraint_description="Add upper structure triads to create richer harmony"
    ))

    # ------------------------------------------------------------------------
    # Bass Line Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="bass_cliche_prob",
        full_path="harmony.progression.bass_cliche_prob",
        description="Probability of bass cliché (chromatic bass line)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["jazz", "standards", "bossa_nova"],
        module_file="generators/harmony_module.py",
        constraint_description="Descending chromatic bass. Classic jazz standard device"
    ))

    REGISTRY.register(ParameterDefinition(
        name="chromatic_bass_motion",
        full_path="harmony.progression.chromatic_bass_motion",
        description="Degree of chromatic bass motion in progressions",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "classical"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates smooth, sophisticated bass lines"
    ))

    REGISTRY.register(ParameterDefinition(
        name="contrary_motion_bass",
        full_path="harmony.progression.contrary_motion_bass",
        description="Preference for contrary motion between bass and melody",
        param_type=ParameterType.PROBABILITY,
        default_value=0.6,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"],
        module_file="generators/harmony_module.py",
        constraint_description="Classical voice leading principle. Creates independence"
    ))


# ============================================================================
# SECTION 4: NEO-RIEMANNIAN & ADVANCED THEORY (14 parameters)
# ============================================================================

def register_neo_riemannian_theory():
    """
    Register 14 Neo-Riemannian and advanced music theory parameters.

    Neo-Riemannian theory analyzes chromatic harmony through transformations
    (P, L, R). Also includes set theory, microtonality, and spectral harmony.
    """

    # ------------------------------------------------------------------------
    # Neo-Riemannian Transformations
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="parallel_transform",
        full_path="harmony.neo_riemannian.parallel_transform",
        description="Probability of Parallel (P) transform (major↔minor, common root)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "film_score"],
        module_file="generators/harmony_module.py",
        constraint_description="C major ↔ C minor. Mode change preserving root"
    ))

    REGISTRY.register(ParameterDefinition(
        name="relative_transform",
        full_path="harmony.neo_riemannian.relative_transform",
        description="Probability of Relative (R) transform (major↔minor, minor 3rd)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic"],
        module_file="generators/harmony_module.py",
        constraint_description="C major ↔ A minor. Traditional relative relationship"
    ))

    REGISTRY.register(ParameterDefinition(
        name="leading_tone_transform",
        full_path="harmony.neo_riemannian.leading_tone_transform",
        description="Probability of Leading-tone (L) transform (major↔minor, semitone)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["classical", "romantic", "progressive"],
        module_file="generators/harmony_module.py",
        constraint_description="C major ↔ E minor. Shares major/minor third"
    ))

    REGISTRY.register(ParameterDefinition(
        name="slide_transform",
        full_path="harmony.neo_riemannian.slide_transform",
        description="Probability of Slide (S) transform (parallel move keeping 3rd/5th)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "progressive"],
        module_file="generators/harmony_module.py",
        constraint_description="C major ↔ C# minor. Creates chromatic motion"
    ))

    REGISTRY.register(ParameterDefinition(
        name="hexatonic_pole_probability",
        full_path="harmony.neo_riemannian.hexatonic_pole_probability",
        description="Probability of hexatonic pole relationships (L-P cycle)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.08,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["romantic", "film_score", "Wagner"],
        module_file="generators/harmony_module.py",
        constraint_description="Maximally distant chords in hexatonic system. Wagner, Brahms"
    ))

    # ------------------------------------------------------------------------
    # Pitch Class Set Theory
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="pitch_class_sets",
        full_path="harmony.set_theory.pitch_class_sets",
        description="Use pitch class set theory for chord construction",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary_classical", "avant_garde", "serial"],
        module_file="generators/harmony_module.py",
        constraint_description="Allen Forte's set theory. 12-tone and atonal music"
    ))

    REGISTRY.register(ParameterDefinition(
        name="interval_vector_preference",
        full_path="harmony.set_theory.interval_vector_preference",
        description="Preferred interval vector [c1,c2,c3,c4,c5,c6] for pc sets",
        param_type=ParameterType.ARRAY_INT,
        default_value=[0, 0, 0, 0, 0, 0],
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["contemporary_classical", "serial"],
        module_file="generators/harmony_module.py",
        constraint_description="Counts interval classes. Used for set class analysis"
    ))

    REGISTRY.register(ParameterDefinition(
        name="z_relation_exploration",
        full_path="harmony.set_theory.z_relation_exploration",
        description="Probability of exploring Z-related sets (same interval vector)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["contemporary_classical", "serial"],
        module_file="generators/harmony_module.py",
        constraint_description="Sets with identical interval vectors but different pitch content"
    ))

    REGISTRY.register(ParameterDefinition(
        name="aggregate_completion",
        full_path="harmony.set_theory.aggregate_completion",
        description="Ensure all 12 pitch classes used before repeating",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["serial", "twelve_tone"],
        module_file="generators/harmony_module.py",
        constraint_description="Serialist technique. Ensures equal importance of all pitches"
    ))

    REGISTRY.register(ParameterDefinition(
        name="combinatoriality",
        full_path="harmony.set_theory.combinatoriality",
        description="Degree of combinatorial relationships in set usage",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["serial", "twelve_tone"],
        module_file="generators/harmony_module.py",
        constraint_description="Sets that combine to form aggregates. Schoenberg, Webern"
    ))

    # ------------------------------------------------------------------------
    # Microtonal Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="quarter_tone_usage",
        full_path="harmony.microtonality.quarter_tone_usage",
        description="Probability of using quarter tones (50 cent intervals)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["contemporary_classical", "middle_eastern", "experimental"],
        module_file="generators/harmony_module.py",
        constraint_description="Divides semitone in half. Arabic maqam, contemporary classical"
    ))

    REGISTRY.register(ParameterDefinition(
        name="just_intonation",
        full_path="harmony.microtonality.just_intonation",
        description="Use just intonation (pure ratios) instead of equal temperament",
        param_type=ParameterType.BOOLEAN,
        default_value=False,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["early_music", "barbershop", "experimental", "spectral"],
        module_file="generators/harmony_module.py",
        constraint_description="Acoustically pure intervals. Used in early music, Harry Partch"
    ))

    REGISTRY.register(ParameterDefinition(
        name="cent_deviation",
        full_path="harmony.microtonality.cent_deviation",
        description="Maximum pitch deviation in cents from equal temperament",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.0,
        min_value=0.0,
        max_value=50.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["experimental", "spectral", "just_intonation"],
        module_file="generators/harmony_module.py",
        constraint_description="Allows microtuning. 0=equal temperament, 50=quarter-tone"
    ))

    # ------------------------------------------------------------------------
    # Spectral Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="harmonic_series_chords",
        full_path="harmony.spectral.harmonic_series_chords",
        description="Probability of using harmonic series-based chords",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["spectral", "contemporary_classical", "electronic"],
        module_file="generators/harmony_module.py",
        constraint_description="Chords based on natural harmonic overtones. Grisey, Murail"
    ))


# ============================================================================
# SECTION 5: SPECIALIZED TECHNIQUES (10 parameters)
# ============================================================================

def register_specialized_techniques():
    """
    Register 10 specialized harmony technique parameters.

    Additional advanced concepts including negative harmony, extended
    tertian harmony, slash chords, and cross-cultural techniques.
    """

    # ------------------------------------------------------------------------
    # Negative Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="negative_harmony_prob",
        full_path="harmony.advanced.negative_harmony_prob",
        description="Probability of negative harmony transformation",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["jazz", "experimental", "contemporary"],
        module_file="generators/harmony_module.py",
        constraint_description="Reflect harmony around tonic-dominant axis. Popularized by Jacob Collier"
    ))

    # ------------------------------------------------------------------------
    # Slash Chords and Inversions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="slash_chord_prob",
        full_path="harmony.advanced.slash_chord_prob",
        description="Probability of slash chords (chord over alternate bass)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["pop", "rock", "gospel", "CCM"],
        module_file="generators/harmony_module.py",
        constraint_description="E.g., C/G, D/F#. Creates bass lines, polychordal effects"
    ))

    REGISTRY.register(ParameterDefinition(
        name="first_inversion_prob",
        full_path="harmony.advanced.first_inversion_prob",
        description="Probability of first inversion chords (third in bass)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "pop", "hymn"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates smooth bass motion. Weaker than root position"
    ))

    REGISTRY.register(ParameterDefinition(
        name="second_inversion_prob",
        full_path="harmony.advanced.second_inversion_prob",
        description="Probability of second inversion chords (fifth in bass)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.15,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.MEDIUM,
        genre_relevance=["classical", "baroque"],
        module_file="generators/harmony_module.py",
        constraint_description="Traditionally requires resolution. Cadential 6/4 most common"
    ))

    # ------------------------------------------------------------------------
    # Extended Tertian Harmony
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="extended_tertian_prob",
        full_path="harmony.advanced.extended_tertian_prob",
        description="Probability of chords with 9th, 11th, and 13th simultaneously",
        param_type=ParameterType.PROBABILITY,
        default_value=0.2,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "fusion", "neo_soul"],
        module_file="generators/harmony_module.py",
        constraint_description="Maximum tertian complexity. C13(11,9) has 7 notes"
    ))

    # ------------------------------------------------------------------------
    # Cross-Cultural Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="ragam_influence",
        full_path="harmony.advanced.ragam_influence",
        description="Degree of Indian ragam/raga influence on harmony",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["indian_classical", "world_fusion", "raga_rock"],
        module_file="generators/harmony_module.py",
        constraint_description="Incorporates vadi, samvadi, and raga-specific phrases"
    ))

    REGISTRY.register(ParameterDefinition(
        name="maqam_influence",
        full_path="harmony.advanced.maqam_influence",
        description="Degree of Arabic maqam influence on harmony",
        param_type=ParameterType.PROBABILITY,
        default_value=0.0,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["middle_eastern", "arabic", "world_fusion"],
        module_file="generators/harmony_module.py",
        constraint_description="Incorporates quarter tones, jins, and maqam progressions"
    ))

    # ------------------------------------------------------------------------
    # Harmonic Series Extensions
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="blue_note_influence",
        full_path="harmony.advanced.blue_note_influence",
        description="Degree of blue note influence (b3, b5, b7 in major context)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.3,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["blues", "jazz", "rock", "soul"],
        module_file="generators/harmony_module.py",
        constraint_description="Creates characteristic blues sound. Microtonal in origin"
    ))

    # ------------------------------------------------------------------------
    # Contemporary Jazz Techniques
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="superimposition_prob",
        full_path="harmony.advanced.superimposition_prob",
        description="Probability of chord superimposition (playing 'outside')",
        param_type=ParameterType.PROBABILITY,
        default_value=0.1,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.HIGH,
        genre_relevance=["jazz", "bebop", "fusion", "free_jazz"],
        module_file="generators/harmony_module.py",
        constraint_description="Temporarily play unrelated chord. Creates tension. Coltrane technique"
    ))

    # ------------------------------------------------------------------------
    # Harmonic Complexity Metric
    # ------------------------------------------------------------------------

    REGISTRY.register(ParameterDefinition(
        name="harmonic_complexity_target",
        full_path="harmony.advanced.harmonic_complexity_target",
        description="Target harmonic complexity (0.0=simple, 1.0=maximum)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        category=ParameterCategory.HARMONY,
        musical_impact=MusicalImpact.CRITICAL,
        genre_relevance=["all"],
        module_file="generators/harmony_module.py",
        constraint_description="Master control balancing simplicity vs. sophistication"
    ))


# ============================================================================
# MAIN REGISTRATION FUNCTION
# ============================================================================

def register_all_harmony_deep_expansion():
    """
    Register all 94 harmony deep expansion parameters.

    This expands the harmony parameter count from 56 to 150 total.

    Breakdown:
    - Advanced Voicing Techniques: 25 parameters
    - Advanced Chord Extensions: 20 parameters
    - Advanced Progressions: 25 parameters
    - Neo-Riemannian & Advanced Theory: 14 parameters
    - Specialized Techniques: 10 parameters

    Total: 94 new parameters
    """

    print("=" * 80)
    print("HARMONY DEEP EXPANSION - Agent 3")
    print("=" * 80)
    print("\nExpanding harmony parameters from 56 to 150...")
    print("\nRegistering parameter groups:")

    # Get initial count
    initial_stats = REGISTRY.get_statistics()
    initial_harmony = len([p for p in REGISTRY.parameters.values()
                           if p.category == ParameterCategory.HARMONY])

    print(f"\n  Initial harmony parameters: {initial_harmony}")

    # Register all sections
    print("\n  [1/5] Advanced Voicing Techniques (25 params)...")
    register_advanced_voicing_techniques()

    print("  [2/5] Advanced Chord Extensions (20 params)...")
    register_advanced_chord_extensions()

    print("  [3/5] Advanced Progressions (25 params)...")
    register_advanced_progressions()

    print("  [4/5] Neo-Riemannian & Advanced Theory (14 params)...")
    register_neo_riemannian_theory()

    print("  [5/5] Specialized Techniques (10 params)...")
    register_specialized_techniques()

    # Get final count
    final_stats = REGISTRY.get_statistics()
    final_harmony = len([p for p in REGISTRY.parameters.values()
                         if p.category == ParameterCategory.HARMONY])

    added = final_harmony - initial_harmony

    print("\n" + "=" * 80)
    print("REGISTRATION COMPLETE")
    print("=" * 80)
    print(f"\n  ✅ Initial harmony parameters: {initial_harmony}")
    print(f"  ✅ New parameters added: {added}")
    print(f"  ✅ Final harmony parameters: {final_harmony}")
    print(f"  ✅ Total system parameters: {final_stats['total_parameters']}")

    # Verify we hit target
    if added == 94:
        print(f"\n  🎯 TARGET ACHIEVED: Added exactly 94 parameters!")
    else:
        print(f"\n  ⚠️  Warning: Expected 94 new parameters, added {added}")

    print("\n" + "=" * 80)

    return final_harmony, added


# ============================================================================
# DOCUMENTATION GENERATION
# ============================================================================

def generate_harmony_expansion_documentation(output_path: str):
    """Generate comprehensive documentation for harmony expansion"""

    harmony_params = [p for p in REGISTRY.parameters.values()
                     if p.category == ParameterCategory.HARMONY]

    lines = []
    lines.append("# Harmony Deep Expansion - Complete Parameter Documentation")
    lines.append("")
    lines.append("**Agent 3: Harmony Deep Expansion Specialist**")
    lines.append("")
    lines.append(f"Total Harmony Parameters: {len(harmony_params)}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Group by section
    sections = {
        "Advanced Voicing Techniques": [
            "harmony.voicing.quartal_probability",
            "harmony.voicing.quintal_probability",
            "harmony.voicing.cluster_probability",
            "harmony.voicing.cluster_width",
            "harmony.voicing.upper_structure_triad",
            "harmony.voicing.polychord_probability",
            "harmony.voicing.bitonal_probability",
            "harmony.voicing.shell_voicing_prob",
            "harmony.voicing.rootless_prob",
            "harmony.voicing.so_what_voicing",
            "harmony.voicing.drop2_prob",
            "harmony.voicing.drop3_prob",
            "harmony.voicing.drop24_prob",
            "harmony.voicing.spread_voicing_prob",
            "harmony.voicing.close_position_prob",
            "harmony.voicing.open_position_prob",
            "harmony.voicing.doubled_root",
            "harmony.voicing.doubled_third",
            "harmony.voicing.doubled_fifth",
            "harmony.voicing.doubled_seventh",
            "harmony.voicing.omit_root",
            "harmony.voicing.omit_fifth",
            "harmony.voicing.guide_tone_emphasis",
            "harmony.voicing.color_tone_emphasis",
            "harmony.voicing.avoid_note_handling",
        ],
        "Advanced Chord Extensions": [
            "harmony.extension.ninth_usage",
            "harmony.extension.flat_ninth_prob",
            "harmony.extension.sharp_ninth_prob",
            "harmony.extension.eleventh_usage",
            "harmony.extension.sharp_eleventh_prob",
            "harmony.extension.thirteenth_usage",
            "harmony.extension.flat_thirteenth_prob",
            "harmony.extension.add2_prob",
            "harmony.extension.add4_prob",
            "harmony.extension.add6_prob",
            "harmony.extension.sus2_prob",
            "harmony.extension.sus4_prob",
            "harmony.extension.augmented_prob",
            "harmony.extension.diminished_prob",
            "harmony.extension.half_diminished_prob",
            "harmony.extension.altered_dominant_prob",
            "harmony.extension.lydian_dominant_prob",
            "harmony.extension.phrygian_dominant_prob",
            "harmony.extension.dominant_bebop_prob",
            "harmony.extension.stacked_intervals",
        ],
        "Advanced Progressions": [
            "harmony.progression.tritone_substitution",
            "harmony.progression.chromatic_mediant",
            "harmony.progression.modal_interchange",
            "harmony.progression.secondary_dominant_prob",
            "harmony.progression.secondary_diminished_prob",
            "harmony.progression.deceptive_cadence_prob",
            "harmony.progression.plagal_cadence_prob",
            "harmony.progression.half_cadence_prob",
            "harmony.progression.turnaround_type",
            "harmony.progression.cycle_movement",
            "harmony.progression.coltrane_changes",
            "harmony.progression.giant_steps_prob",
            "harmony.progression.backdoor_progression",
            "harmony.progression.pedal_point_harmony",
            "harmony.progression.static_harmony_duration",
            "harmony.progression.harmonic_rhythm",
            "harmony.progression.chords_per_bar",
            "harmony.progression.reharmonization_prob",
            "harmony.progression.polytonality",
            "harmony.progression.parallel_harmony",
            "harmony.progression.planing_type",
            "harmony.progression.upper_structure_reharmonization",
            "harmony.progression.bass_cliche_prob",
            "harmony.progression.chromatic_bass_motion",
            "harmony.progression.contrary_motion_bass",
        ],
        "Neo-Riemannian & Advanced Theory": [
            "harmony.neo_riemannian.parallel_transform",
            "harmony.neo_riemannian.relative_transform",
            "harmony.neo_riemannian.leading_tone_transform",
            "harmony.neo_riemannian.slide_transform",
            "harmony.neo_riemannian.hexatonic_pole_probability",
            "harmony.set_theory.pitch_class_sets",
            "harmony.set_theory.interval_vector_preference",
            "harmony.set_theory.z_relation_exploration",
            "harmony.set_theory.aggregate_completion",
            "harmony.set_theory.combinatoriality",
            "harmony.microtonality.quarter_tone_usage",
            "harmony.microtonality.just_intonation",
            "harmony.microtonality.cent_deviation",
            "harmony.spectral.harmonic_series_chords",
        ],
        "Specialized Techniques": [
            "harmony.advanced.negative_harmony_prob",
            "harmony.advanced.slash_chord_prob",
            "harmony.advanced.first_inversion_prob",
            "harmony.advanced.second_inversion_prob",
            "harmony.advanced.extended_tertian_prob",
            "harmony.advanced.ragam_influence",
            "harmony.advanced.maqam_influence",
            "harmony.advanced.blue_note_influence",
            "harmony.advanced.superimposition_prob",
            "harmony.advanced.harmonic_complexity_target",
        ]
    }

    for section_name, param_paths in sections.items():
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| Parameter | Type | Default | Range/Options | Description |")
        lines.append("|-----------|------|---------|---------------|-------------|")

        for path in param_paths:
            param = REGISTRY.get(path)
            if param:
                type_str = param.param_type.value
                default_str = str(param.default_value)

                if param.param_type == ParameterType.CATEGORICAL:
                    range_str = f"{param.options}"
                elif param.min_value is not None and param.max_value is not None:
                    range_str = f"[{param.min_value}, {param.max_value}]"
                elif param.param_type == ParameterType.PROBABILITY:
                    range_str = "[0.0, 1.0]"
                else:
                    range_str = "-"

                desc_short = param.description[:80] + "..." if len(param.description) > 80 else param.description

                lines.append(f"| `{path}` | {type_str} | {default_str} | {range_str} | {desc_short} |")

        lines.append("")

    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n📄 Documentation written to: {output_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("HARMONY DEEP EXPANSION MODULE")
    print("Agent 3: Harmony Deep Expansion Specialist")
    print("=" * 80)

    # Register all parameters
    final_count, added_count = register_all_harmony_deep_expansion()

    # Generate documentation
    doc_path = "/home/user/Do/midi_generator/parameters/HARMONY_EXPANSION.md"
    generate_harmony_expansion_documentation(doc_path)

    # Display some statistics
    print("\n📊 Parameter Statistics by Musical Impact:")
    harmony_params = [p for p in REGISTRY.parameters.values()
                     if p.category == ParameterCategory.HARMONY]

    impact_counts = {}
    for param in harmony_params:
        impact = param.musical_impact.value
        impact_counts[impact] = impact_counts.get(impact, 0) + 1

    for impact, count in sorted(impact_counts.items()):
        print(f"  {impact:15s}: {count:3d} parameters")

    print("\n📊 Most Relevant Genres:")
    genre_counts = {}
    for param in harmony_params:
        for genre in param.genre_relevance:
            genre_counts[genre] = genre_counts.get(genre, 0) + 1

    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for genre, count in top_genres:
        print(f"  {genre:25s}: {count:3d} parameters")

    print("\n" + "=" * 80)
    print("HARMONY DEEP EXPANSION COMPLETE")
    print("=" * 80)
    print("\n✅ Module ready for integration with XGBoost learning system")
    print("✅ All parameters are independently learnable")
    print("✅ Musical theory foundations documented")
    print("✅ Genre relevance mapped for all parameters")
    print("\n" + "=" * 80 + "\n")
