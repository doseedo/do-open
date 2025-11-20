#!/usr/bin/env python3
"""
Modern Big Band Style Profiles
================================

This module implements style profiles for modern big band arrangers:
- Thad Jones (1923-1986): Angular melodies, quartal harmony, wide intervals
- Maria Schneider (1960-): Orchestral colors, impressionistic, cinematic
- Gordon Goodwin (1954-): High energy, contemporary swing, complex rhythms

These profiles define arranging characteristics that can be applied to generate
authentic-sounding arrangements in the style of these master arrangers.

Research Sources:
----------------
- Thad Jones scores: "A Child is Born", "Three and One", "The Deacon"
- Maria Schneider: "Concert in the Garden", orchestration masterclasses
- Gordon Goodwin: "Hunting Wabbits", Big Phat Band recordings
- Modern jazz harmony: Quartal/quintal voicings, altered scales, metric modulation

Author: Agent 15 - Modern Big Band Style Analyzer
Date: 2025
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import random


# ==============================================================================
# MODERN HARMONY TYPES
# ==============================================================================

class ModernHarmonyType(Enum):
    """Modern jazz harmony techniques."""
    QUARTAL = "quartal"                    # Stacked 4ths (McCoy Tyner style)
    QUINTAL = "quintal"                    # Stacked 5ths
    CLUSTER = "cluster"                    # Tight chromatic clusters
    POLYCHORD = "polychord"                # Two chords stacked
    UPPER_STRUCTURE = "upper_structure"     # Triads over bass note
    SLASH_CHORD = "slash_chord"            # Triad over different bass
    SUSPENDED = "suspended"                # Sus chords (sus2, sus4)
    ALTERED = "altered"                    # Highly altered dominants


class VoicingSpacing(Enum):
    """Voicing spacing preferences."""
    CLOSE = "close"                        # All voices within octave
    OPEN = "open"                          # Spread within 1.5 octaves
    WIDE = "wide"                          # Spread across 2+ octaves
    MIXED = "mixed"                        # Varies by register
    DROP_2 = "drop_2"                      # Drop 2nd voice from top
    DROP_3 = "drop_3"                      # Drop 3rd voice from top


# ==============================================================================
# STYLE PROFILE DATA CLASS
# ==============================================================================

@dataclass
class StyleProfile:
    """
    Complete style profile for a big band arranger.

    Defines all characteristics needed to generate arrangements in a specific style.
    """
    # Identity
    name: str
    era: str                               # "modern", "contemporary", "postmodern"

    # Orchestration
    voicing_preference: str                # Primary voicing type
    voicing_spacing: str                   # Spacing preference
    voicing_variety: float = 0.5           # 0-1: How much to vary voicings

    # Harmony
    harmony_complexity: float = 0.5        # 0-1: Harmonic sophistication
    use_quartal: float = 0.0              # 0-1: Quartal harmony usage
    use_clusters: float = 0.0             # 0-1: Cluster chord usage
    use_polychords: float = 0.0           # 0-1: Polychord usage
    use_altered_dominants: float = 0.3    # 0-1: Altered dominant usage
    chord_extensions: List[int] = field(default_factory=lambda: [7, 9])

    # Melody
    melodic_intervals: str = "mixed"       # "stepwise", "angular", "mixed"
    angular_melodies: bool = False         # Wide interval jumps
    use_chromatic: float = 0.3            # 0-1: Chromatic approach notes
    phrase_length_variance: float = 0.3    # 0-1: Phrase length variation

    # Rhythm
    rhythmic_complexity: float = 0.5       # 0-1: Rhythmic sophistication
    use_odd_meters: float = 0.0           # 0-1: 5/4, 7/4, etc.
    use_metric_modulation: bool = False    # Tempo relationships
    syncopation_level: float = 0.5        # 0-1: Amount of syncopation

    # Articulation
    articulation_variety: float = 0.5      # 0-1: Variety of articulations
    use_falls: float = 0.2                # 0-1: Fall usage
    use_doits: float = 0.1                # 0-1: Doit usage
    use_shakes: float = 0.1               # 0-1: Shake usage

    # Dynamics
    dynamic_range: str = "medium"          # "narrow", "medium", "wide", "very_wide"
    use_crescendo: float = 0.5            # 0-1: Crescendo usage
    terraced_dynamics: bool = False        # Sudden dynamic shifts

    # Texture
    texture_density: str = "medium"        # "sparse", "medium", "dense", "varied"
    use_unison: float = 0.2               # 0-1: Unison passages
    use_tutti: float = 0.3                # 0-1: Full ensemble
    section_contrast: float = 0.5          # 0-1: Contrast between sections

    # Form
    intro_style: str = "vamp"              # "rubato", "vamp", "ostinato", "full"
    ending_style: str = "tag"              # "tag", "fermata", "fade", "abrupt"
    use_interludes: float = 0.2           # 0-1: Interlude sections

    # Special Techniques
    woodwind_doublings: bool = False       # Flute, clarinet doubles
    use_pedal_tones: float = 0.2          # 0-1: Pedal point usage
    impressionistic: bool = False          # Impressionistic colors
    use_ostinato: float = 0.2             # 0-1: Repeated patterns

    # Tempo
    typical_tempo_range: Tuple[int, int] = (80, 200)  # BPM range
    tempo_variation: bool = False          # Within-piece tempo changes

    # Additional characteristics
    characteristics: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# THAD JONES STYLE PROFILE
# ==============================================================================

THAD_JONES_STYLE = StyleProfile(
    name="Thad Jones",
    era="modern",

    # Orchestration - Modern voicings with wide spacing
    voicing_preference="quartal_and_spread",
    voicing_spacing="wide",
    voicing_variety=0.8,  # High variety

    # Harmony - Complex, modern jazz harmony
    harmony_complexity=0.85,
    use_quartal=0.6,  # Signature: stacked 4ths
    use_clusters=0.3,
    use_polychords=0.2,
    use_altered_dominants=0.7,
    chord_extensions=[7, 9, 11, 13],  # Rich extensions

    # Melody - Angular, wide intervals
    melodic_intervals="angular",
    angular_melodies=True,
    use_chromatic=0.5,
    phrase_length_variance=0.6,  # Irregular phrases

    # Rhythm - Complex, sophisticated
    rhythmic_complexity=0.8,
    use_odd_meters=0.3,  # Occasional 5/4, 7/4
    use_metric_modulation=False,
    syncopation_level=0.7,

    # Articulation - Varied
    articulation_variety=0.7,
    use_falls=0.4,
    use_doits=0.2,
    use_shakes=0.3,

    # Dynamics - Wide range for expression
    dynamic_range="wide",
    use_crescendo=0.7,
    terraced_dynamics=False,

    # Texture - Rich, full, but with contrast
    texture_density="varied",
    use_unison=0.2,
    use_tutti=0.4,
    section_contrast=0.7,  # High contrast between sections

    # Form
    intro_style="ostinato",
    ending_style="fermata",
    use_interludes=0.3,

    # Special Techniques
    woodwind_doublings=False,
    use_pedal_tones=0.3,
    impressionistic=False,
    use_ostinato=0.4,

    # Tempo - Moderate to fast
    typical_tempo_range=(100, 220),
    tempo_variation=False,

    characteristics={
        "style_keywords": ["angular", "modern", "sophisticated", "wide_intervals"],
        "influences": ["bebop", "post-bop", "modal"],
        "signature_techniques": ["quartal_voicings", "wide_spacing", "complex_harmony"],
        "typical_forms": ["aaba", "blues", "through_composed"],
        "mood": "sophisticated, modern, intellectual",
    }
)


# ==============================================================================
# MARIA SCHNEIDER STYLE PROFILE
# ==============================================================================

MARIA_SCHNEIDER_STYLE = StyleProfile(
    name="Maria Schneider",
    era="contemporary",

    # Orchestration - Orchestral colors, unique timbres
    voicing_preference="orchestral_colors",
    voicing_spacing="mixed",  # Varies by context
    voicing_variety=0.9,  # Very high variety

    # Harmony - Very complex, impressionistic
    harmony_complexity=0.9,
    use_quartal=0.5,
    use_clusters=0.4,  # Color clusters
    use_polychords=0.3,
    use_altered_dominants=0.6,
    chord_extensions=[7, 9, 11, 13],

    # Melody - Lyrical, flowing
    melodic_intervals="mixed",
    angular_melodies=False,
    use_chromatic=0.4,
    phrase_length_variance=0.7,  # Very irregular, organic

    # Rhythm - Moderate complexity
    rhythmic_complexity=0.6,
    use_odd_meters=0.2,
    use_metric_modulation=False,
    syncopation_level=0.4,

    # Articulation - Very nuanced
    articulation_variety=0.9,  # Highest variety
    use_falls=0.3,
    use_doits=0.2,
    use_shakes=0.2,

    # Dynamics - Very wide range (cinematic)
    dynamic_range="very_wide",
    use_crescendo=0.8,  # Frequent builds
    terraced_dynamics=False,

    # Texture - Highly varied (sparse to dense)
    texture_density="varied",
    use_unison=0.1,  # Rare - prefers harmony
    use_tutti=0.3,
    section_contrast=0.8,  # High contrast

    # Form
    intro_style="rubato",  # Often free, atmospheric
    ending_style="fade",  # Gradual endings
    use_interludes=0.5,  # Frequent interludes

    # Special Techniques - SIGNATURE
    woodwind_doublings=True,  # SIGNATURE: Flute, clarinet doubles
    use_pedal_tones=0.7,  # SIGNATURE: Frequent pedals
    impressionistic=True,  # SIGNATURE: Impressionistic colors
    use_ostinato=0.5,  # Repeated patterns for texture

    # Tempo - Slow to moderate (ballads to medium)
    typical_tempo_range=(60, 140),
    tempo_variation=True,  # Rubato sections

    characteristics={
        "style_keywords": ["impressionistic", "cinematic", "orchestral", "atmospheric"],
        "influences": ["Gil Evans", "impressionism", "film music"],
        "signature_techniques": ["woodwind_doublings", "pedal_tones", "layered_textures"],
        "typical_forms": ["through_composed", "suite_form", "multi_movement"],
        "mood": "atmospheric, cinematic, evocative",
    }
)


# ==============================================================================
# GORDON GOODWIN STYLE PROFILE
# ==============================================================================

GORDON_GOODWIN_STYLE = StyleProfile(
    name="Gordon Goodwin",
    era="contemporary",

    # Orchestration - Powerful, contemporary
    voicing_preference="powerful_contemporary",
    voicing_spacing="open",
    voicing_variety=0.6,

    # Harmony - Modern but accessible
    harmony_complexity=0.7,
    use_quartal=0.4,
    use_clusters=0.2,
    use_polychords=0.2,
    use_altered_dominants=0.7,
    chord_extensions=[7, 9, 11, 13],

    # Melody - Catchy, energetic
    melodic_intervals="mixed",
    angular_melodies=False,
    use_chromatic=0.5,
    phrase_length_variance=0.4,

    # Rhythm - Very complex, high energy
    rhythmic_complexity=0.9,  # SIGNATURE: Complex rhythms
    use_odd_meters=0.4,
    use_metric_modulation=True,  # Occasional
    syncopation_level=0.9,  # VERY syncopated

    # Articulation - Punchy, precise
    articulation_variety=0.6,
    use_falls=0.4,
    use_doits=0.3,
    use_shakes=0.2,

    # Dynamics - Wide, but powerful
    dynamic_range="wide",
    use_crescendo=0.6,
    terraced_dynamics=True,  # Sudden shifts

    # Texture - Dense, powerful
    texture_density="dense",
    use_unison=0.4,  # Powerful unisons
    use_tutti=0.6,  # Frequent full ensemble
    section_contrast=0.6,

    # Form
    intro_style="full",  # Bold intros
    ending_style="abrupt",  # Strong endings
    use_interludes=0.2,

    # Special Techniques
    woodwind_doublings=False,
    use_pedal_tones=0.3,
    impressionistic=False,
    use_ostinato=0.4,

    # Tempo - Fast! High energy
    typical_tempo_range=(160, 260),  # SIGNATURE: Fast tempos
    tempo_variation=False,

    characteristics={
        "style_keywords": ["energetic", "powerful", "contemporary", "complex_rhythms"],
        "influences": ["bebop", "latin_jazz", "funk", "fusion"],
        "signature_techniques": ["fast_tempos", "complex_rhythms", "powerful_tutti"],
        "typical_forms": ["aaba", "latin_forms", "through_composed"],
        "mood": "energetic, exciting, virtuosic",
    }
)


# ==============================================================================
# STYLE PROFILE UTILITIES
# ==============================================================================

def get_style_profile(name: str) -> Optional[StyleProfile]:
    """
    Get a style profile by name.

    Args:
        name: Style name ("thad_jones", "maria_schneider", "gordon_goodwin")

    Returns:
        StyleProfile or None if not found
    """
    profiles = {
        "thad_jones": THAD_JONES_STYLE,
        "thad": THAD_JONES_STYLE,
        "maria_schneider": MARIA_SCHNEIDER_STYLE,
        "schneider": MARIA_SCHNEIDER_STYLE,
        "maria": MARIA_SCHNEIDER_STYLE,
        "gordon_goodwin": GORDON_GOODWIN_STYLE,
        "goodwin": GORDON_GOODWIN_STYLE,
        "gordon": GORDON_GOODWIN_STYLE,
    }
    return profiles.get(name.lower())


def list_available_styles() -> List[str]:
    """List all available modern big band styles."""
    return ["thad_jones", "maria_schneider", "gordon_goodwin"]


# ==============================================================================
# MODERN BIG BAND ARRANGER
# ==============================================================================

class ModernBigBandArranger:
    """
    Modern big band arranger implementing contemporary techniques.

    This class applies modern arranging techniques based on style profiles:
    - Quartal/quintal voicings
    - Wide interval voicings
    - Complex rhythmic patterns
    - Contemporary harmonic language
    - Textural variety

    Usage:
        arranger = ModernBigBandArranger(style_profile=THAD_JONES_STYLE)
        arrangement = arranger.arrange(melody, chords, form)
    """

    def __init__(self, style_profile: StyleProfile):
        """
        Initialize arranger with a style profile.

        Args:
            style_profile: StyleProfile (e.g., THAD_JONES_STYLE)
        """
        self.style = style_profile

    def arrange(self, melody, chords, form=None):
        """
        Create modern big band arrangement.

        Args:
            melody: List of melody notes
            chords: List of chord events
            form: Musical form (optional)

        Returns:
            Dict with arrangement tracks
        """
        # This is a placeholder for the full implementation
        # The actual arrangement logic would be implemented here

        arrangement = {
            'style': self.style.name,
            'melody': melody,
            'chords': chords,
            'voicing_type': self.style.voicing_preference,
            'characteristics': self.style.characteristics,
        }

        return arrangement

    def generate_quartal_voicing(self, root_note: int, num_voices: int = 4) -> List[int]:
        """
        Generate quartal (stacked 4ths) voicing.

        This is a signature technique of modern arrangers like Thad Jones
        and McCoy Tyner.

        Args:
            root_note: Base MIDI note
            num_voices: Number of voices (default 4)

        Returns:
            List of MIDI note numbers
        """
        voicing = []
        current = root_note

        for i in range(num_voices):
            voicing.append(current)
            # Perfect 4th = 5 semitones
            current += 5

            # Add slight variation (sometimes augmented 4th)
            if self.style.voicing_variety > 0.7 and random.random() < 0.3:
                current += 1  # Make it augmented 4th (6 semitones total)

        return voicing

    def generate_wide_spacing_voicing(self, chord_tones: List[int],
                                     min_spacing: int = 7) -> List[int]:
        """
        Generate wide-spaced voicing (Thad Jones technique).

        Args:
            chord_tones: Chord tone MIDI notes
            min_spacing: Minimum spacing between adjacent voices (semitones)

        Returns:
            Wide-spaced voicing
        """
        if len(chord_tones) < 2:
            return chord_tones

        voicing = [chord_tones[0]]  # Start with bass note

        for i in range(1, len(chord_tones)):
            # Ensure minimum spacing
            next_note = chord_tones[i]

            while next_note - voicing[-1] < min_spacing:
                next_note += 12  # Raise an octave

            voicing.append(next_note)

        return voicing

    def apply_dynamic_shape(self, notes, section_type: str = "a_section"):
        """
        Apply dynamic shaping based on style profile.

        Args:
            notes: List of notes to shape
            section_type: Section type for context

        Returns:
            Notes with applied dynamics
        """
        # Dynamic ranges based on profile
        dynamic_ranges = {
            "narrow": (60, 90),
            "medium": (50, 100),
            "wide": (40, 110),
            "very_wide": (30, 120),
        }

        min_vel, max_vel = dynamic_ranges.get(
            self.style.dynamic_range,
            (50, 100)
        )

        # Apply crescendo if style uses it
        if random.random() < self.style.use_crescendo:
            # Apply crescendo over the phrase
            for i, note in enumerate(notes):
                progress = i / max(len(notes) - 1, 1)
                velocity = int(min_vel + (max_vel - min_vel) * progress)
                note['velocity'] = max(1, min(127, velocity))

        return notes

    def suggest_intro_type(self) -> str:
        """
        Suggest intro type based on style profile.

        Returns:
            Intro style string
        """
        return self.style.intro_style

    def suggest_ending_type(self) -> str:
        """
        Suggest ending type based on style profile.

        Returns:
            Ending style string
        """
        return self.style.ending_style

    def get_typical_tempo(self) -> int:
        """
        Get typical tempo for this style.

        Returns:
            Tempo in BPM
        """
        min_tempo, max_tempo = self.style.typical_tempo_range
        return random.randint(min_tempo, max_tempo)


# ==============================================================================
# VALIDATION AND COMPARISON
# ==============================================================================

def compare_style_characteristics():
    """
    Compare characteristics of the three modern arrangers.

    This helps understand the differences between styles.
    """
    styles = [THAD_JONES_STYLE, MARIA_SCHNEIDER_STYLE, GORDON_GOODWIN_STYLE]

    print("MODERN BIG BAND STYLE COMPARISON")
    print("=" * 80)
    print()

    characteristics = [
        ("Harmony Complexity", "harmony_complexity"),
        ("Quartal Usage", "use_quartal"),
        ("Rhythmic Complexity", "rhythmic_complexity"),
        ("Articulation Variety", "articulation_variety"),
        ("Dynamic Range", "dynamic_range"),
        ("Texture Density", "texture_density"),
        ("Typical Tempo Range", "typical_tempo_range"),
    ]

    for char_name, char_attr in characteristics:
        print(f"{char_name}:")
        for style in styles:
            value = getattr(style, char_attr)
            print(f"  {style.name:20s}: {value}")
        print()

    # Special techniques
    print("Special Techniques:")
    for style in styles:
        techniques = style.characteristics.get("signature_techniques", [])
        print(f"  {style.name:20s}: {', '.join(techniques)}")
    print()

    # Mood/Feel
    print("Mood/Feel:")
    for style in styles:
        mood = style.characteristics.get("mood", "")
        print(f"  {style.name:20s}: {mood}")
    print()


if __name__ == "__main__":
    # Run comparison when module is run directly
    compare_style_characteristics()

    print("\nExample: Creating Thad Jones arranger")
    arranger = ModernBigBandArranger(THAD_JONES_STYLE)
    print(f"Arranger created for: {arranger.style.name}")
    print(f"Suggested intro: {arranger.suggest_intro_type()}")
    print(f"Suggested ending: {arranger.suggest_ending_type()}")
    print(f"Typical tempo: {arranger.get_typical_tempo()} BPM")

    print("\nExample: Quartal voicing from C (MIDI 60)")
    voicing = arranger.generate_quartal_voicing(60, 4)
    print(f"Quartal voicing: {voicing}")
    print(f"Intervals: {[voicing[i+1] - voicing[i] for i in range(len(voicing)-1)]}")
