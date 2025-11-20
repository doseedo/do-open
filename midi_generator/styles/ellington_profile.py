#!/usr/bin/env python3
"""
Duke Ellington Style Profile
=============================

Comprehensive style profile for Duke Ellington's arranging approach.

Duke Ellington (1899-1974) was one of the most influential composers and arrangers
in jazz history. His sophisticated orchestration, exotic harmonies, and unique
tonal colors revolutionized big band music.

Key Characteristics:
--------------------
1. **Plunger Mute Brass**: Signature sound (Bubber Miley, Cootie Williams)
   - Wa-wa effects, growls, jungle sounds
   - Used extensively in brass section (60% of notes)

2. **Exotic Harmonies**: Whole tone, diminished, bitonal
   - Extended chords (9ths, 11ths, 13ths)
   - Parallel motion and unconventional progressions
   - Modal mixture and chromatic harmony

3. **Voice as Instrument**: Unusual doublings
   - Clarinet + muted trombone (unique timbre)
   - Alto sax + muted trumpet
   - Creative instrument combinations

4. **Jungle Sounds**: Growls, rips, falls
   - High articulation variety (80%)
   - Expressive pitch bends and glissandi

5. **Rich Harmony**: Complex voicings
   - Close voicings with unusual doublings
   - Varied spacing (not uniform)
   - Dense harmonic texture

6. **Wide Dynamic Range**: ppp to fff
   - Dramatic contrasts
   - Frequent crescendos and diminuendos

7. **Sophisticated Form**: Complex structures
   - Often free-form intros (rubato)
   - Extended forms beyond 32-bar AABA
   - Through-composed sections

Famous Works Referenced:
------------------------
- "Ko-Ko" (1940) - Plunger mutes, growls, jungle style
- "Caravan" (1936) - Exotic harmony, sustained brass pads
- "Mood Indigo" (1930) - Unusual orchestration, clarinet on bottom
- "Concerto for Cootie" (1940) - Plunger mute virtuosity
- "Black and Tan Fantasy" (1927) - Growl techniques
- "Harlem Airshaft" (1940) - Complex layered textures

Research Sources:
-----------------
- Mark Tucker: "The Duke Ellington Reader"
- Gunther Schuller: "The Swing Era" (Chapter on Ellington)
- Living Jazz Archives: livingjazzarchives.org
- eJazzLines: ejazzlines.com
- Duke Ellington Music Society: depanorama.net

Author: Agent 13 - Duke Ellington Style Analyzer
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class VoicingPreference(Enum):
    """Voicing style preferences."""
    CLOSE = "close"
    DROP_2 = "drop_2"
    DROP_3 = "drop_3"
    SPREAD = "spread"
    CLOSE_WITH_DOUBLINGS = "close_with_doublings"  # Ellington signature
    VARIED = "varied"  # Mix of different voicings


class DynamicRange(Enum):
    """Dynamic range options."""
    NARROW = "narrow"       # Limited dynamics
    MEDIUM = "medium"       # Normal range
    WIDE = "wide"          # Professional range
    VERY_WIDE = "very_wide"  # Extreme contrasts (Ellington)


class TextureDensity(Enum):
    """Texture density options."""
    SPARSE = "sparse"      # Minimal instrumentation
    MEDIUM = "medium"      # Balanced
    RICH = "rich"         # Full, dense
    VARIED = "varied"     # Changes throughout


@dataclass
class EllingtonStyleConfig:
    """
    Complete style configuration for Duke Ellington arrangements.

    This comprehensive profile captures the essence of Ellington's
    arranging style based on analysis of his major works.
    """

    # === ORCHESTRATION ===
    voicing_preference: VoicingPreference = VoicingPreference.CLOSE_WITH_DOUBLINGS
    voicing_spacing: str = "varied"  # Not uniform spacing
    use_plunger_mutes: float = 0.6  # 60% of brass notes
    use_growls: float = 0.4  # 40% probability
    unusual_doublings: bool = True  # Signature Ellington technique

    # === HARMONY ===
    harmony_complexity: float = 0.9  # Very complex (0.0-1.0 scale)
    use_whole_tone: float = 0.3  # Whole tone scale usage
    use_diminished: float = 0.4  # Diminished scale/chords
    use_bitonal: float = 0.2  # Bitonal harmony (occasional)
    chord_extensions: List[int] = field(default_factory=lambda: [9, 11, 13])  # Rich extensions
    use_parallel_motion: float = 0.3  # Parallel voicings

    # === ARTICULATIONS ===
    articulation_variety: float = 0.8  # High variety (0.0-1.0)
    fall_probability: float = 0.6  # High usage of falls
    shake_probability: float = 0.3  # Moderate shakes
    rip_probability: float = 0.4  # Moderate rips
    scoop_probability: float = 0.3  # Moderate scoops
    doit_probability: float = 0.3  # Moderate doits

    # === DYNAMICS ===
    dynamic_range: DynamicRange = DynamicRange.VERY_WIDE
    use_crescendo: float = 0.7  # Frequent crescendos
    use_diminuendo: float = 0.6  # Frequent diminuendos
    min_velocity: int = 25  # ppp
    max_velocity: int = 127  # fff

    # === FORM ===
    intro_style: str = "rubato"  # Free tempo intro (common)
    ending_style: str = "fermata"  # Sustained ending
    use_extended_forms: bool = True  # Beyond 32-bar AABA
    through_composed: float = 0.3  # Sometimes through-composed

    # === TEXTURE ===
    texture_density: TextureDensity = TextureDensity.RICH
    layering_complexity: float = 0.8  # Multiple simultaneous layers
    use_countermelodies: float = 0.6  # Frequent counterpoint

    # === RHYTHM ===
    swing_ratio: float = 0.62  # Standard medium swing
    use_rubato: float = 0.3  # Occasional rubato sections
    rhythmic_complexity: float = 0.7  # Moderately complex rhythms

    # === INSTRUMENTAL BALANCE ===
    brass_prominence: float = 0.7  # Prominent brass
    sax_prominence: float = 0.8  # Prominent saxes
    rhythm_section_prominence: float = 0.6  # Supporting role
    use_clarinet: float = 0.5  # Occasional clarinet solos/doublings

    # === SPECIFIC TECHNIQUES ===
    jungle_sounds: bool = True  # Signature early Ellington
    wa_wa_effects: float = 0.5  # Plunger mute wa-wa
    exotic_scales: float = 0.4  # Whole tone, diminished, etc.
    chromatic_harmony: float = 0.6  # Chromatic voice leading

    # === MELODIC CHARACTERISTICS ===
    melodic_complexity: float = 0.7  # Sophisticated melodies
    use_blue_notes: float = 0.6  # Blues inflections
    wide_intervals: float = 0.5  # Some angular motion

    # === PERFORMANCE PRACTICE ===
    expressive_timing: float = 0.6  # Rubato, timing variation
    vibrato_intensity: float = 0.6  # Moderate to strong vibrato
    portamento: float = 0.4  # Glissandi between notes


# Pre-configured Ellington style profile
ELLINGTON_STYLE = EllingtonStyleConfig()


# === STYLE COMPARISON DATA ===
# For reference and validation

ELLINGTON_VS_BASIE = {
    "ellington": {
        "harmony_complexity": 0.9,
        "texture_density": 0.8,
        "articulation_variety": 0.8,
        "exotic_harmony": 0.6,
        "description": "Complex, exotic, orchestral colors"
    },
    "basie": {
        "harmony_complexity": 0.3,
        "texture_density": 0.5,
        "articulation_variety": 0.4,
        "exotic_harmony": 0.1,
        "description": "Simple, riff-based, rhythm-driven"
    }
}


# === ELLINGTON SIGNATURE TECHNIQUES ===

ELLINGTON_DOUBLINGS = [
    {
        "combination": ["clarinet", "muted_trombone"],
        "register": "low",
        "effect": "Unique dark timbre",
        "example": "Mood Indigo"
    },
    {
        "combination": ["alto_sax", "muted_trumpet"],
        "register": "middle",
        "effect": "Blended color",
        "example": "Caravan"
    },
    {
        "combination": ["baritone_sax", "bass_trombone"],
        "register": "low",
        "effect": "Rich bass foundation",
        "example": "Ko-Ko"
    },
]


ELLINGTON_HARMONIC_DEVICES = {
    "whole_tone": {
        "usage": "Over dominant 7th chords",
        "effect": "Exotic, impressionistic sound",
        "examples": ["Take the A Train", "Sophisticated Lady"]
    },
    "diminished": {
        "usage": "Passing chords, dominants",
        "effect": "Tension, chromaticism",
        "examples": ["Ko-Ko", "Harlem Airshaft"]
    },
    "bitonal": {
        "usage": "Occasional polytonal sections",
        "effect": "Dissonant, modern sound",
        "examples": ["The Mooche"]
    },
    "parallel_motion": {
        "usage": "Moving voices in parallel",
        "effect": "Thick, impressionistic texture",
        "examples": ["Mood Indigo", "Prelude to a Kiss"]
    }
}


ELLINGTON_ARTICULATIONS = {
    "plunger_mute": {
        "technique": "Open/closed positions creating wa-wa",
        "primary_users": ["Bubber Miley", "Cootie Williams", "Ray Nance"],
        "frequency": 0.6,
        "effect": "Talking, vocal quality"
    },
    "growl": {
        "technique": "Singing/humming while playing",
        "primary_users": ["Bubber Miley", "Tricky Sam Nanton"],
        "frequency": 0.4,
        "effect": "Jungle sounds, raw expression"
    },
    "fall": {
        "technique": "Pitch bend downward at phrase end",
        "frequency": 0.6,
        "typical_range": "-200 to -400 cents",
        "effect": "Expressive ending, blues feeling"
    },
    "shake": {
        "technique": "Rapid lip trill on sustained notes",
        "frequency": 0.3,
        "effect": "Excitement, emphasis"
    }
}


# === DYNAMIC CONTOURS ===

ELLINGTON_DYNAMIC_PROFILES = {
    "dramatic_contrast": {
        "description": "Extreme dynamic shifts",
        "min_velocity": 25,  # ppp
        "max_velocity": 127,  # fff
        "contrast_range": 102,
        "usage": "Emotional impact, drama"
    },
    "gradual_build": {
        "description": "Long crescendo to climax",
        "start": 50,  # p
        "peak": 115,  # ff
        "duration_bars": 16,
        "usage": "Building tension"
    },
    "terraced_dynamics": {
        "description": "Sudden level changes",
        "levels": [40, 75, 105],  # p, mf, f
        "usage": "Sectional contrast"
    }
}


# === FORM STRUCTURES ===

ELLINGTON_FORMS = {
    "standard_32_bar": {
        "structure": "AABA",
        "frequency": 0.4,
        "notes": "Traditional form but with sophisticated harmony"
    },
    "extended_form": {
        "structure": "AABACDA",
        "frequency": 0.3,
        "notes": "Longer, more complex forms"
    },
    "through_composed": {
        "structure": "Non-repeating",
        "frequency": 0.2,
        "notes": "Suites, tone poems (Black, Brown and Beige)"
    },
    "free_intro": {
        "duration": "4-8 bars",
        "style": "Rubato, impressionistic",
        "frequency": 0.5,
        "notes": "Sets mood before tempo"
    }
}


# === VOICING EXAMPLES ===

ELLINGTON_VOICINGS = {
    "mood_indigo": {
        "instrumentation": ["clarinet", "muted_trumpet", "muted_trombone"],
        "register": "low to middle",
        "spacing": "close",
        "effect": "Unique dark timbre",
        "innovation": "Clarinet on bottom (unusual)"
    },
    "ko_ko": {
        "instrumentation": ["full_brass", "saxes"],
        "voicing_type": "close_with_doublings",
        "special": "Plunger brass throughout",
        "effect": "Jungle, primitive sound"
    },
    "sophisticated_lady": {
        "harmony": "Extended chords (9ths, 11ths, 13ths)",
        "voicing_type": "rich_close",
        "melody_instrument": "alto_sax",
        "effect": "Lush, romantic"
    }
}


def get_ellington_style() -> EllingtonStyleConfig:
    """
    Get Duke Ellington style configuration.

    Returns:
        EllingtonStyleConfig: Complete Ellington style profile
    """
    return ELLINGTON_STYLE


def compare_to_basie() -> Dict[str, Any]:
    """
    Compare Ellington and Basie styles.

    Useful for understanding the dramatic differences between
    these two giants of big band music.

    Returns:
        Dict: Comparison data
    """
    return ELLINGTON_VS_BASIE


def get_signature_techniques() -> Dict[str, Any]:
    """
    Get Ellington's signature arranging techniques.

    Returns:
        Dict: Technique descriptions and usage data
    """
    return {
        "doublings": ELLINGTON_DOUBLINGS,
        "harmonic_devices": ELLINGTON_HARMONIC_DEVICES,
        "articulations": ELLINGTON_ARTICULATIONS,
        "dynamic_profiles": ELLINGTON_DYNAMIC_PROFILES,
        "forms": ELLINGTON_FORMS,
        "voicing_examples": ELLINGTON_VOICINGS
    }


if __name__ == "__main__":
    # Demo: Print Ellington style profile
    print("=" * 80)
    print("DUKE ELLINGTON STYLE PROFILE")
    print("=" * 80)
    print()

    style = get_ellington_style()

    print("ORCHESTRATION:")
    print(f"  Voicing Preference: {style.voicing_preference.value}")
    print(f"  Plunger Mutes: {style.use_plunger_mutes * 100:.0f}%")
    print(f"  Growls: {style.use_growls * 100:.0f}%")
    print(f"  Unusual Doublings: {style.unusual_doublings}")
    print()

    print("HARMONY:")
    print(f"  Complexity: {style.harmony_complexity * 100:.0f}%")
    print(f"  Whole Tone Usage: {style.use_whole_tone * 100:.0f}%")
    print(f"  Diminished Usage: {style.use_diminished * 100:.0f}%")
    print(f"  Bitonal Usage: {style.use_bitonal * 100:.0f}%")
    print(f"  Extensions: {style.chord_extensions}")
    print()

    print("ARTICULATIONS:")
    print(f"  Variety: {style.articulation_variety * 100:.0f}%")
    print(f"  Falls: {style.fall_probability * 100:.0f}%")
    print(f"  Shakes: {style.shake_probability * 100:.0f}%")
    print(f"  Rips: {style.rip_probability * 100:.0f}%")
    print()

    print("DYNAMICS:")
    print(f"  Range: {style.dynamic_range.value}")
    print(f"  Crescendo Usage: {style.use_crescendo * 100:.0f}%")
    print(f"  Velocity Range: {style.min_velocity}-{style.max_velocity}")
    print()

    print("FORM:")
    print(f"  Intro Style: {style.intro_style}")
    print(f"  Ending Style: {style.ending_style}")
    print(f"  Extended Forms: {style.use_extended_forms}")
    print()

    print("TEXTURE:")
    print(f"  Density: {style.texture_density.value}")
    print(f"  Layering: {style.layering_complexity * 100:.0f}%")
    print(f"  Countermelodies: {style.use_countermelodies * 100:.0f}%")
    print()

    print("ELLINGTON VS. BASIE:")
    comparison = compare_to_basie()
    for composer, data in comparison.items():
        print(f"\n  {composer.upper()}:")
        print(f"    {data['description']}")
        print(f"    Harmony: {data['harmony_complexity'] * 100:.0f}%")
        print(f"    Texture: {data['texture_density'] * 100:.0f}%")

    print()
    print("=" * 80)
