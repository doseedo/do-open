"""
Duke Ellington Style Profile

Based on Duke Ellington's arranging principles:
- Complex, exotic harmonies (whole tone, diminished, bitonal)
- Rich orchestral colors
- Plunger mute brass (Bubber Miley, Cootie Williams signature)
- Growls and "jungle" sounds
- Unusual doublings (clarinet + muted trombone)
- Wide dynamic range (ppp to fff)
- Rich chord extensions (9ths, 11ths, 13ths)
- Varied voicings (not consistent spacing like Basie)

Reference recordings: "Ko-Ko", "Caravan", "Mood Indigo", "Concerto for Cootie"
"""

from .base_profile import StyleProfile


ELLINGTON_STYLE = StyleProfile(
    # ============== METADATA ==============
    style_name="Duke Ellington",
    description="Complex harmony with exotic orchestral colors and plunger mutes",
    composer_reference="Duke Ellington Orchestra",
    era="swing",
    typical_tempo_range=(80, 160),  # Wide range - ballads to swingers

    # ============== ORCHESTRATION ==============
    voicing_preference="close_with_doublings",  # Close with unique doublings
    voicing_spacing="varied",            # NOT consistent - varied for color
    use_section_hits=0.5,                # Moderate
    use_riffs=0.4,                       # Less than Basie
    use_plunger_mutes=0.6,              # HIGH - SIGNATURE SOUND
    use_growls=0.4,                     # MODERATE-HIGH - jungle sounds
    unusual_doublings=True,              # SIGNATURE - clarinet + trombone, etc.

    # ============== HARMONY ==============
    harmony_complexity=0.9,              # VERY COMPLEX
    chord_extensions=[9, 11, 13],        # RICH HARMONY
    use_blues=0.5,                      # Some blues, but diverse
    use_whole_tone=0.3,                 # MODERATE - exotic sound
    use_diminished=0.4,                 # MODERATE-HIGH
    use_bitonal=0.2,                    # OCCASIONAL - very advanced
    use_modal_interchange=0.4,          # Moderate
    use_tritone_subs=0.5,               # Frequent
    reharmonization_level=0.7,          # High complexity

    # ============== PIANO ==============
    piano_style="comping",               # Regular comping, not sparse
    piano_density=0.6,                  # Moderate density
    use_stride=True,                    # Sometimes uses stride
    rootless_voicings=True,             # Modern voicings

    # ============== RHYTHM SECTION ==============
    emphasis_on_rhythm=0.6,              # Balanced - horns also important
    feathered_kick=False,                # Not Ellington's signature
    walking_bass_style="bebop",          # More modern bass lines
    freddie_green_guitar=False,          # Not the Ellington sound

    # ============== ARTICULATIONS ==============
    articulation_variety=0.8,            # HIGH VARIETY
    fall_probability=0.6,                # FREQUENT falls
    shake_probability=0.3,              # Moderate shakes
    staccato_probability=0.4,            # Mix of articulations
    preferred_articulations=["fall_long", "growl", "shake", "plunger"],

    # ============== DYNAMICS ==============
    dynamic_range="wide",                # PPP to FFF
    use_crescendo=0.7,                  # Frequent dynamic shaping
    shout_chorus_intensity=0.85,        # Strong but not Basie-level

    # ============== FORM ==============
    intro_style="rubato",                # Often free, rubato intros
    ending_style="fermata",             # Sustained endings
    use_modulation=True,                # Sometimes modulates

    # ============== TEXTURE ==============
    texture_density=0.8,                 # RICH, FULL TEXTURE
    riff_based=False,                   # Not riff-based - through-composed
    call_response=0.5,                   # Some call-response

    # ============== SWING FEEL ==============
    swing_ratio=0.62,                    # Medium swing
    swing_intensity=0.9,                 # Slightly looser than Basie
    laid_back_feel=0.0,                  # Neutral
)
