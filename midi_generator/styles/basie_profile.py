#!/usr/bin/env python3
"""
Count Basie Style Profile
==========================

This module defines the arranging style characteristics of Count Basie and
his legendary big band. Basie's style is characterized by:

**Key Characteristics:**
1. **Simplicity** - "Head" arrangements, easy to customize
2. **Riff-Based** - Short, repeated rhythmic figures
3. **Sparse Piano** - Minimalist comping (opposite of stride)
4. **Powerful Rhythm Section** - Freddie Green guitar (4-to-the-bar), feathered kick
5. **Section Hits** - Punchy brass/sax stabs
6. **Shout Chorus** - Famous climactic sections
7. **"Button" Endings** - Short, punchy conclusions
8. **Blues Foundation** - Strong blues influence
9. **Swing Feel** - Infectious, driving swing

**Basie vs. Ellington:**
- Basie: Simple, riff-based, rhythm section driven
- Ellington: Complex, orchestral colors, exotic harmony

**Famous Recordings for Reference:**
- "One O'Clock Jump" - Classic shout chorus
- "April in Paris" - Famous shout chorus ending
- "Li'l Darlin'" - Slow, sparse, beautiful
- "Corner Pocket" - Riff-based swing
- "Jumpin' at the Woodside" - Powerful swing

**Research Sources:**
- ejazzlines.com transcriptions
- "The Basie Way" - analysis of arranging techniques
- Freddie Green rhythm guitar style
- Jo Jones drumming (feathered bass drum)
- Count Basie minimalist piano style

Author: Agent 14 - Count Basie Style Analyzer
Date: 2025
License: MIT
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum


class BasieTextureStyle(Enum):
    """Texture styles used in Basie arrangements."""
    SPARSE = "sparse"           # Minimal, space between figures
    RIFF_BASED = "riff_based"   # Repeated short figures
    SHOUT = "shout"             # Full band, climactic
    SECTION_HITS = "hits"       # Punchy stabs


class BasiePianoStyle(Enum):
    """Piano comping styles."""
    SPARSE = "sparse"           # Minimal, strategic comps
    STRIDE = "stride"           # Occasionally used
    BLOCK_CHORDS = "block"      # Rare, for emphasis
    SILENT = "silent"           # Often lays out


class BasieRiffType(Enum):
    """Types of riffs in Basie style."""
    BLUES_RIFF = "blues"        # Blues-based figures
    CALL_RESPONSE = "call_response"  # Sections trade
    BACKGROUND = "background"   # Behind solos
    SHOUT = "shout"            # Climactic riffs


@dataclass
class BasieStyleConfig:
    """
    Complete style configuration for Count Basie arrangements.

    This configuration can be passed to arrangers to generate
    authentic Basie-style arrangements.
    """

    # ========== ORCHESTRATION ==========
    voicing_preference: str = "unison_and_octaves"  # Simple voicings
    voicing_spacing: str = "open"  # Spread voicings for power
    use_section_hits: float = 0.9  # Very high - signature Basie
    use_riffs: float = 0.8  # Riff-based arrangements

    # ========== HARMONY ==========
    harmony_complexity: float = 0.3  # Simple, functional harmony
    use_blues: float = 0.7  # Blues-based progressions
    chord_extensions: List[int] = field(default_factory=lambda: [7])  # Basic 7th chords
    avoid_complex_extensions: bool = True  # No 9/11/13 typically

    # ========== PIANO ==========
    piano_style: str = "sparse"  # Minimalist comping
    piano_density: float = 0.2  # Very sparse (20% of possible comps)
    piano_silence_probability: float = 0.3  # Often lays out
    piano_accent_on_hits: bool = True  # Emphasize section hits

    # ========== RHYTHM SECTION ==========
    emphasis_on_rhythm: float = 0.9  # Rhythm section is the star
    feathered_kick: bool = True  # All four beats, soft (signature)
    freddie_green_guitar: bool = True  # 4-to-the-bar rhythm guitar
    hi_hat_on_2_and_4: bool = True  # Classic swing

    # ========== ARTICULATIONS ==========
    articulation_variety: float = 0.4  # Less than Ellington
    staccato_probability: float = 0.7  # Punchy, crisp
    accent_on_hits: float = 0.9  # Strong accents on section hits
    fall_probability: float = 0.3  # Some falls, not excessive

    # ========== DYNAMICS ==========
    dynamic_range: str = "medium"  # Less extreme than Ellington
    shout_chorus_intensity: float = 1.0  # Famous powerful shout choruses
    base_velocity: int = 85  # Medium-loud base
    shout_velocity: int = 105  # Loud shout chorus
    sparse_section_velocity: int = 70  # Softer for contrast

    # ========== FORM ==========
    intro_style: str = "vamp"  # Simple vamp or button
    ending_style: str = "button"  # Short punchy ending (signature)
    use_shout_chorus: bool = True  # Almost always
    shout_chorus_location: str = "final_A"  # Last A in AABA

    # ========== TEXTURE ==========
    texture_density: float = 0.5  # Sparser than Ellington (0.8)
    riff_based: bool = True  # Signature characteristic
    space_between_phrases: bool = True  # Let music breathe
    background_density: float = 0.4  # Sparse backgrounds behind solos

    # ========== SECTION WRITING ==========
    sax_section_style: str = "unison_or_octaves"  # Simple
    brass_section_style: str = "hits_and_riffs"  # Punchy
    section_hits_per_chorus: int = 8  # Frequent hits
    riff_length_bars: int = 2  # Short, repeatable riffs

    # ========== SWING FEEL ==========
    swing_ratio: float = 0.64  # Medium swing
    swing_consistency: bool = True  # Very consistent swing
    groove_pocket: str = "deep"  # In the pocket
    tempo_preference: str = "medium_up"  # 140-200 BPM sweet spot

    # ========== BLUES CHARACTERISTICS ==========
    blues_influence: float = 0.8  # Strong blues foundation
    use_blues_scale: bool = True  # Frequent blues scale usage
    blue_notes: bool = True  # b3, b5, b7
    shuffle_feel_occasional: bool = True  # Sometimes shuffle

    # ========== SPECIFIC TECHNIQUES ==========
    basie_button_intro: bool = True  # Short punchy intro
    basie_button_ending: bool = True  # Short punchy ending
    basie_piano_stabs: bool = True  # Strategic piano punctuation
    one_note_piano: bool = True  # Famous Basie minimalism
    half_valve_effects: bool = False  # Rare in Basie vs. Ellington

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            'voicing_preference': self.voicing_preference,
            'voicing_spacing': self.voicing_spacing,
            'use_section_hits': self.use_section_hits,
            'use_riffs': self.use_riffs,
            'harmony_complexity': self.harmony_complexity,
            'use_blues': self.use_blues,
            'chord_extensions': self.chord_extensions,
            'piano_style': self.piano_style,
            'piano_density': self.piano_density,
            'emphasis_on_rhythm': self.emphasis_on_rhythm,
            'feathered_kick': self.feathered_kick,
            'freddie_green_guitar': self.freddie_green_guitar,
            'articulation_variety': self.articulation_variety,
            'staccato_probability': self.staccato_probability,
            'dynamic_range': self.dynamic_range,
            'shout_chorus_intensity': self.shout_chorus_intensity,
            'intro_style': self.intro_style,
            'ending_style': self.ending_style,
            'texture_density': self.texture_density,
            'riff_based': self.riff_based,
            'swing_ratio': self.swing_ratio,
            'blues_influence': self.blues_influence,
        }


# ============================================================================
# DEFAULT BASIE STYLE PROFILE
# ============================================================================

BASIE_STYLE = BasieStyleConfig()

# Preset variations for different Basie eras/contexts

BASIE_EARLY_KANSAS_CITY = BasieStyleConfig(
    blues_influence=0.9,
    riff_based=True,
    piano_density=0.3,  # Slightly more active early on
    tempo_preference="medium_up",
)

BASIE_1950s_ATOMIC = BasieStyleConfig(
    # "Atomic Mr. Basie" era - peak precision
    section_hits_per_chorus=12,
    piano_density=0.15,  # Even more minimal
    swing_consistency=True,
    groove_pocket="deep",
)

BASIE_BALLAD = BasieStyleConfig(
    # "Li'l Darlin'" style
    tempo_preference="slow",
    texture_density=0.3,
    piano_density=0.1,
    space_between_phrases=True,
    base_velocity=65,
    shout_velocity=85,
)

BASIE_SHOUT_CHORUS_ONLY = BasieStyleConfig(
    # Maximum intensity (final chorus)
    shout_chorus_intensity=1.0,
    section_hits_per_chorus=16,
    texture_density=0.9,
    base_velocity=100,
    shout_velocity=115,
)


# ============================================================================
# COMPARISON PROFILES (for validation and contrast)
# ============================================================================

# Simplified Ellington profile for contrast
ELLINGTON_COMPARISON = {
    "voicing_preference": "close_with_doublings",
    "harmony_complexity": 0.9,  # vs. Basie 0.3
    "piano_density": 0.7,  # vs. Basie 0.2
    "texture_density": 0.8,  # vs. Basie 0.5
    "articulation_variety": 0.8,  # vs. Basie 0.4
    "use_exotic_harmony": 0.7,
    "plunger_mutes": 0.6,
}


# ============================================================================
# VALIDATION METRICS
# ============================================================================

BASIE_VALIDATION_CRITERIA = {
    "piano_sparseness": {
        "metric": "note_density",
        "target": 0.2,
        "tolerance": 0.1,
        "description": "Piano should be very sparse (Basie minimalism)"
    },
    "section_hits": {
        "metric": "hits_per_chorus",
        "target": 8,
        "tolerance": 3,
        "description": "Frequent punchy section hits"
    },
    "harmony_simplicity": {
        "metric": "average_chord_extensions",
        "target": 1.0,  # Mostly 7th chords
        "tolerance": 0.5,
        "description": "Simple harmony (7th chords, not 9/11/13)"
    },
    "swing_ratio": {
        "metric": "swing_ratio",
        "target": 0.64,
        "tolerance": 0.02,
        "description": "Medium swing feel"
    },
    "shout_chorus_intensity": {
        "metric": "final_chorus_velocity_increase",
        "target": 20,  # 20 velocity points higher
        "tolerance": 5,
        "description": "Shout chorus significantly louder"
    },
    "blues_content": {
        "metric": "blues_scale_usage",
        "target": 0.7,
        "tolerance": 0.15,
        "description": "Strong blues influence"
    },
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_basie_style_for_context(context: str) -> BasieStyleConfig:
    """
    Get appropriate Basie style config for different contexts.

    Args:
        context: One of 'standard', 'early', 'atomic', 'ballad', 'shout'

    Returns:
        Appropriate BasieStyleConfig
    """
    styles = {
        'standard': BASIE_STYLE,
        'early': BASIE_EARLY_KANSAS_CITY,
        'atomic': BASIE_1950s_ATOMIC,
        'ballad': BASIE_BALLAD,
        'shout': BASIE_SHOUT_CHORUS_ONLY,
    }

    return styles.get(context, BASIE_STYLE)


def validate_basie_characteristics(arrangement: Dict) -> Dict[str, bool]:
    """
    Validate that an arrangement has authentic Basie characteristics.

    Args:
        arrangement: Dictionary with arrangement data

    Returns:
        Dictionary of validation results
    """
    # This would analyze the arrangement and check against validation criteria
    # Placeholder for now - would be implemented with actual analysis
    return {
        "piano_sparseness": True,
        "section_hits_present": True,
        "simple_harmony": True,
        "swing_feel": True,
        "shout_chorus": True,
        "overall_authentic": True,
    }


if __name__ == "__main__":
    # Demo: print Basie style characteristics
    print("COUNT BASIE STYLE PROFILE")
    print("=" * 80)
    print()

    basie = BASIE_STYLE

    print("ORCHESTRATION:")
    print(f"  Voicing: {basie.voicing_preference}")
    print(f"  Section Hits: {basie.use_section_hits * 100}%")
    print(f"  Riff-Based: {basie.use_riffs * 100}%")
    print()

    print("HARMONY:")
    print(f"  Complexity: {basie.harmony_complexity * 100}% (simple)")
    print(f"  Blues Influence: {basie.use_blues * 100}%")
    print(f"  Chord Extensions: {basie.chord_extensions}")
    print()

    print("PIANO:")
    print(f"  Style: {basie.piano_style}")
    print(f"  Density: {basie.piano_density * 100}% (very sparse)")
    print()

    print("RHYTHM SECTION:")
    print(f"  Emphasis: {basie.emphasis_on_rhythm * 100}%")
    print(f"  Feathered Kick: {basie.feathered_kick}")
    print(f"  Freddie Green Guitar: {basie.freddie_green_guitar}")
    print()

    print("DYNAMICS:")
    print(f"  Range: {basie.dynamic_range}")
    print(f"  Shout Chorus Intensity: {basie.shout_chorus_intensity * 100}%")
    print()

    print("FORM:")
    print(f"  Intro: {basie.intro_style}")
    print(f"  Ending: {basie.ending_style}")
    print()

    print("TEXTURE:")
    print(f"  Density: {basie.texture_density * 100}%")
    print(f"  Riff-Based: {basie.riff_based}")
    print()

    print("=" * 80)
    print("\nStyle profile loaded successfully!")
    print("Use BASIE_STYLE for standard Basie arrangements")
    print("Use get_basie_style_for_context('ballad') for ballad style, etc.")
