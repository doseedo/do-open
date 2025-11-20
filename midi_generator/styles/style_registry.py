#!/usr/bin/env python3
"""
Style Profile Registry - Composer and Performer Styles

Provides style profiles that define how different composers, arrangers, and performers
approach music. These profiles can be applied to any ensemble to generate music "in the
style of" a particular artist.

Examples:
- Count Basie: Simple, riff-based, punchy
- Duke Ellington: Complex, exotic harmonies, orchestral colors
- Mozart: Elegant, balanced, transparent
- Ravi Shankar: Raga-based, improvisational, microtonal

Author: Agent 19 - Genre Scalability Architect
Date: 2025-11-20
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class StyleProfile:
    """
    Configuration for a specific musical style or composer.

    This profile defines aesthetic choices that transcend specific ensembles.
    The same profile can be applied to big band, orchestra, or any ensemble.
    """
    name: str
    composer_era: str  # "baroque", "classical", "romantic", "modern", "jazz_swing", etc.
    cultural_origin: str = "western"  # "western", "indian", "african", "middle_eastern", etc.

    # Orchestration preferences
    voicing_preference: str = "balanced"  # "close", "spread", "balanced", "mixed", "unison", etc.
    voicing_spacing: str = "medium"       # "tight", "medium", "wide"
    doubling_rules: Dict[str, float] = field(default_factory=dict)  # Probabilities

    # Harmonic characteristics
    harmony_complexity: float = 0.5  # 0.0 (simple) to 1.0 (complex)
    chord_extensions: List[int] = field(default_factory=lambda: [7])  # [7, 9, 11, 13]
    chromaticism: float = 0.3  # 0.0 (diatonic) to 1.0 (highly chromatic)
    modulation_frequency: float = 0.2  # 0.0 (no modulation) to 1.0 (frequent)

    # Articulation characteristics
    articulation_variety: float = 0.5  # 0.0 (monotone) to 1.0 (highly varied)
    articulation_probabilities: Dict[str, float] = field(default_factory=dict)
    use_ornamentation: float = 0.3  # 0.0 (none) to 1.0 (heavily ornamented)

    # Dynamic characteristics
    dynamic_range: str = "medium"  # "narrow", "medium", "wide", "very_wide"
    use_crescendo: float = 0.5  # 0.0 (static dynamics) to 1.0 (frequent crescendi)
    sudden_dynamic_changes: float = 0.2  # Subito piano, forte, etc.

    # Rhythmic characteristics
    rhythmic_complexity: float = 0.5  # 0.0 (simple) to 1.0 (complex polyrhythms)
    syncopation: float = 0.3  # 0.0 (none) to 1.0 (heavy syncopation)
    swing_factor: float = 0.0  # 0.0 (straight) to 0.67 (heavy swing)
    rubato_tendency: float = 0.1  # 0.0 (strict tempo) to 1.0 (free tempo)

    # Form preferences
    intro_style: str = "standard"  # "fanfare", "rubato", "vamp", "quiet", "none"
    ending_style: str = "standard"  # "fade", "fermata", "tag", "abrupt", "rall"
    form_adherence: float = 0.8  # 0.0 (free form) to 1.0 (strict form)

    # Texture characteristics
    texture_density: float = 0.5  # 0.0 (sparse) to 1.0 (dense)
    texture_variation: float = 0.5  # 0.0 (static) to 1.0 (highly varied)
    counterpoint_usage: float = 0.3  # 0.0 (homophonic) to 1.0 (polyphonic)

    # Special characteristics
    special_techniques: List[str] = field(default_factory=list)
    avoid_techniques: List[str] = field(default_factory=list)
    signature_sounds: List[str] = field(default_factory=list)


# ==============================================================================
# JAZZ STYLES
# ==============================================================================

BASIE_STYLE = StyleProfile(
    name="Count Basie",
    composer_era="jazz_swing",
    cultural_origin="western",

    # Orchestration - Simple and powerful
    voicing_preference="unison_and_octaves",
    voicing_spacing="open",
    doubling_rules={
        "saxes_unison": 0.6,
        "brass_octaves": 0.5,
        "section_hits": 0.9
    },

    # Harmony - Simple, functional
    harmony_complexity=0.3,
    chord_extensions=[7],  # Basic 7th chords, no fancy extensions
    chromaticism=0.2,
    modulation_frequency=0.1,

    # Articulation - Crisp and punchy
    articulation_variety=0.4,
    articulation_probabilities={
        "staccato": 0.7,
        "accent": 0.5,
        "fall_short": 0.3
    },
    use_ornamentation=0.2,

    # Dynamics - Moderate range, not extreme
    dynamic_range="medium",
    use_crescendo=0.3,
    sudden_dynamic_changes=0.4,  # Basie "shout chorus" hits

    # Rhythm - Swinging, driving
    rhythmic_complexity=0.4,
    syncopation=0.5,
    swing_factor=0.62,  # Medium swing
    rubato_tendency=0.0,  # Strict tempo

    # Form - Simple structures
    intro_style="vamp",
    ending_style="button",  # Famous Basie "button" endings
    form_adherence=0.7,

    # Texture - Sparse, clear
    texture_density=0.5,
    texture_variation=0.4,
    counterpoint_usage=0.2,

    # Special
    signature_sounds=["sparse_piano_comping", "section_hits", "shout_chorus"],
    special_techniques=["feathered_kick_drum", "freddie_green_guitar"]
)


ELLINGTON_STYLE = StyleProfile(
    name="Duke Ellington",
    composer_era="jazz_swing",
    cultural_origin="western",

    # Orchestration - Complex, colorful
    voicing_preference="close_with_doublings",
    voicing_spacing="varied",
    doubling_rules={
        "clarinet_muted_trombone": 0.5,  # Signature doubling
        "unusual_combinations": 0.6
    },

    # Harmony - Rich and exotic
    harmony_complexity=0.9,
    chord_extensions=[9, 11, 13],
    chromaticism=0.6,
    modulation_frequency=0.4,

    # Articulation - Highly varied
    articulation_variety=0.8,
    articulation_probabilities={
        "plunger_mute": 0.6,
        "growl": 0.4,
        "fall": 0.6,
        "shake": 0.3
    },
    use_ornamentation=0.7,

    # Dynamics - Wide range
    dynamic_range="very_wide",
    use_crescendo=0.7,
    sudden_dynamic_changes=0.5,

    # Rhythm - Sophisticated
    rhythmic_complexity=0.6,
    syncopation=0.6,
    swing_factor=0.64,
    rubato_tendency=0.3,  # Rubato in intros

    # Form - Varied and sophisticated
    intro_style="rubato",
    ending_style="fermata",
    form_adherence=0.6,  # More flexible

    # Texture - Rich and dense
    texture_density=0.8,
    texture_variation=0.7,
    counterpoint_usage=0.4,

    # Special
    signature_sounds=["jungle_sounds", "plunger_brass", "exotic_harmonies"],
    special_techniques=["whole_tone_scales", "diminished_scales", "bitonal_voicings"]
)


THAD_JONES_STYLE = StyleProfile(
    name="Thad Jones",
    composer_era="modern_jazz",
    cultural_origin="western",

    # Orchestration - Modern, angular
    voicing_preference="quartal_and_clusters",
    voicing_spacing="wide_intervals",
    doubling_rules={
        "quartal_voicing": 0.6,
        "wide_spacing": 0.7
    },

    # Harmony - Complex and modern
    harmony_complexity=0.8,
    chord_extensions=[9, 11, 13],
    chromaticism=0.7,
    modulation_frequency=0.5,

    # Articulation - Sophisticated
    articulation_variety=0.7,
    articulation_probabilities={
        "accent": 0.6,
        "staccato": 0.5,
        "legato": 0.4
    },
    use_ornamentation=0.5,

    # Dynamics - Wide, expressive
    dynamic_range="wide",
    use_crescendo=0.6,
    sudden_dynamic_changes=0.4,

    # Rhythm - Complex
    rhythmic_complexity=0.8,
    syncopation=0.7,
    swing_factor=0.60,  # Modern, lighter swing
    rubato_tendency=0.2,

    # Form - Modern structures
    intro_style="standard",
    ending_style="standard",
    form_adherence=0.7,

    # Texture - Varied density
    texture_density=0.6,
    texture_variation=0.8,
    counterpoint_usage=0.6,

    # Special
    signature_sounds=["quartal_harmony", "angular_melodies", "modern_voicings"],
    special_techniques=["metric_modulation", "odd_meters"]
)


# ==============================================================================
# CLASSICAL STYLES
# ==============================================================================

MOZART_STYLE = StyleProfile(
    name="Wolfgang Amadeus Mozart",
    composer_era="classical",
    cultural_origin="western",

    # Orchestration - Balanced and elegant
    voicing_preference="balanced",
    voicing_spacing="medium",
    doubling_rules={
        "strings_tutti": 0.7,
        "winds_solo": 0.6,
        "horn_bassoon_doubling": 0.4
    },

    # Harmony - Clear and functional
    harmony_complexity=0.4,
    chord_extensions=[],  # Triads and 7ths, no extensions
    chromaticism=0.3,
    modulation_frequency=0.4,

    # Articulation - Graceful variety
    articulation_variety=0.6,
    articulation_probabilities={
        "staccato": 0.5,
        "legato": 0.4,
        "tenuto": 0.2
    },
    use_ornamentation=0.5,  # Classical trills and turns

    # Dynamics - Moderate, elegant
    dynamic_range="medium",
    use_crescendo=0.4,
    sudden_dynamic_changes=0.2,

    # Rhythm - Clear and structured
    rhythmic_complexity=0.4,
    syncopation=0.2,
    swing_factor=0.0,  # No swing in classical
    rubato_tendency=0.1,

    # Form - Strict adherence
    intro_style="fanfare",
    ending_style="authentic_cadence",
    form_adherence=0.95,  # Very strict

    # Texture - Clear and transparent
    texture_density=0.6,
    texture_variation=0.7,
    counterpoint_usage=0.5,

    # Special
    signature_sounds=["alberti_bass", "singing_melodies", "operatic_influence"],
    special_techniques=["sonata_form", "development_sections"]
)


BEETHOVEN_STYLE = StyleProfile(
    name="Ludwig van Beethoven",
    composer_era="romantic_early",
    cultural_origin="western",

    # Orchestration - Powerful and dramatic
    voicing_preference="spread",
    voicing_spacing="wide",
    doubling_rules={
        "tutti_fortissimo": 0.8,
        "octave_doubling": 0.7
    },

    # Harmony - Innovative for his time
    harmony_complexity=0.6,
    chord_extensions=[7],
    chromaticism=0.5,
    modulation_frequency=0.6,

    # Articulation - Dramatic contrasts
    articulation_variety=0.7,
    articulation_probabilities={
        "marcato": 0.7,
        "sforzando": 0.6,
        "accent": 0.7
    },
    use_ornamentation=0.3,

    # Dynamics - Extreme contrasts
    dynamic_range="very_wide",
    use_crescendo=0.8,
    sudden_dynamic_changes=0.8,  # Signature subito piano/forte

    # Rhythm - Driving, intense
    rhythmic_complexity=0.6,
    syncopation=0.5,
    swing_factor=0.0,
    rubato_tendency=0.2,

    # Form - Expanded classical forms
    intro_style="dramatic",
    ending_style="triumphant",
    form_adherence=0.8,

    # Texture - Varied, dramatic
    texture_density=0.7,
    texture_variation=0.8,
    counterpoint_usage=0.6,

    # Special
    signature_sounds=["fate_motive", "heroic_brass", "dramatic_contrasts"],
    special_techniques=["motivic_development", "scherzo"]
)


# ==============================================================================
# WORLD MUSIC STYLES
# ==============================================================================

RAVI_SHANKAR_STYLE = StyleProfile(
    name="Ravi Shankar",
    composer_era="contemporary",
    cultural_origin="indian",

    # Orchestration - Monophonic melody with drone
    voicing_preference="monophonic",
    voicing_spacing="n/a",
    doubling_rules={
        "tanpura_drone": 1.0,
        "tabla_accompaniment": 1.0
    },

    # Harmony - Modal, not chordal
    harmony_complexity=0.2,  # Simple drone
    chord_extensions=[],  # No chords in traditional sense
    chromaticism=0.8,  # Microtonal inflections
    modulation_frequency=0.0,  # Stays in one raga

    # Articulation - Highly ornamented
    articulation_variety=0.9,
    articulation_probabilities={
        "meend": 0.8,    # Glides
        "gamak": 0.7,    # Oscillations
        "kan": 0.6       # Grace notes
    },
    use_ornamentation=0.95,  # Extremely high

    # Dynamics - Gradual, expressive
    dynamic_range="wide",
    use_crescendo=0.6,
    sudden_dynamic_changes=0.1,

    # Rhythm - Cyclical (tala)
    rhythmic_complexity=0.9,
    syncopation=0.7,
    swing_factor=0.0,
    rubato_tendency=0.8,  # Free in alap, strict in gat

    # Form - Raga structure
    intro_style="alap",  # Slow, unmeasured introduction
    ending_style="jhala",  # Fast rhythmic conclusion
    form_adherence=0.9,  # Strict raga structure

    # Texture - Solo with accompaniment
    texture_density=0.3,
    texture_variation=0.6,
    counterpoint_usage=0.0,  # Monophonic

    # Special
    signature_sounds=["tanpura_drone", "tabla_rhythms", "sitar_technique"],
    special_techniques=["raga_improvisation", "tala_cycles", "microtonal_inflection"]
)


# ==============================================================================
# STYLE REGISTRY
# ==============================================================================

STYLE_REGISTRY: Dict[str, StyleProfile] = {
    # Jazz
    "basie": BASIE_STYLE,
    "ellington": ELLINGTON_STYLE,
    "thad_jones": THAD_JONES_STYLE,

    # Classical
    "mozart": MOZART_STYLE,
    "beethoven": BEETHOVEN_STYLE,

    # World
    "ravi_shankar": RAVI_SHANKAR_STYLE,
}


# ==============================================================================
# REGISTRY FUNCTIONS
# ==============================================================================

def get_style(style_name: str) -> Optional[StyleProfile]:
    """Get style profile by name."""
    return STYLE_REGISTRY.get(style_name.lower())


def register_style(style_name: str, profile: StyleProfile):
    """Register a new style profile."""
    STYLE_REGISTRY[style_name.lower()] = profile


def list_styles() -> List[str]:
    """List all registered styles."""
    return list(STYLE_REGISTRY.keys())


def list_styles_by_era(era: str) -> List[str]:
    """List styles from a specific era."""
    return [
        name for name, profile in STYLE_REGISTRY.items()
        if profile.composer_era == era
    ]


def list_styles_by_culture(culture: str) -> List[str]:
    """List styles from a specific culture."""
    return [
        name for name, profile in STYLE_REGISTRY.items()
        if profile.cultural_origin == culture
    ]


if __name__ == "__main__":
    # Test the registry
    print("Registered Styles:")
    for name in list_styles():
        style = get_style(name)
        print(f"\n{style.name}:")
        print(f"  Era: {style.composer_era}")
        print(f"  Culture: {style.cultural_origin}")
        print(f"  Harmony Complexity: {style.harmony_complexity}")
        print(f"  Voicing: {style.voicing_preference}")
        print(f"  Dynamic Range: {style.dynamic_range}")
