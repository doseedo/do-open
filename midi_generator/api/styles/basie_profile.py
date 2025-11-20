"""
Count Basie Style Profile

Based on Count Basie's arranging principles:
- Simple, riff-based arrangements
- Powerful rhythm section (Freddie Green guitar, feathered kick drum)
- Punchy section hits
- Sparse piano comping (Basie's minimalist style)
- Open, spread voicings
- Famous shout choruses
- "Button" intros and endings
- Blues-based harmony

Reference recordings: "One O'Clock Jump", "April in Paris", "Li'l Darlin'", "Corner Pocket"
"""

from .base_profile import StyleProfile


BASIE_STYLE = StyleProfile(
    # ============== METADATA ==============
    style_name="Count Basie",
    description="Simple, riff-based big band with powerful rhythm section",
    composer_reference="Count Basie Orchestra",
    era="swing",
    typical_tempo_range=(120, 180),

    # ============== ORCHESTRATION ==============
    voicing_preference="open",          # Spread voicings, not clustered
    voicing_spacing="wide",             # Open spacing for powerful sound
    use_section_hits=0.9,               # VERY HIGH - signature Basie sound
    use_riffs=0.8,                      # High - riff-based arrangements
    use_plunger_mutes=0.1,              # Occasional, not signature
    use_growls=0.1,                     # Minimal
    unusual_doublings=False,             # Simple, straightforward

    # ============== HARMONY ==============
    harmony_complexity=0.3,              # SIMPLE - functional harmony
    chord_extensions=[7],                # Basic 7th chords, not complex
    use_blues=0.7,                      # HIGH - blues-based
    use_whole_tone=0.0,                 # Not Basie's style
    use_diminished=0.1,                 # Minimal
    use_bitonal=0.0,                    # No
    use_modal_interchange=0.1,          # Rare
    use_tritone_subs=0.2,               # Some, but not heavy
    reharmonization_level=0.2,          # Keep it simple

    # ============== PIANO ==============
    piano_style="sparse",                # BASIE'S SIGNATURE - minimalist
    piano_density=0.2,                  # VERY SPARSE - famous for space
    use_stride=False,                   # No - opposite of Basie's style
    rootless_voicings=False,            # Simple shell voicings

    # ============== RHYTHM SECTION ==============
    emphasis_on_rhythm=0.9,              # VERY HIGH - rhythm section is star
    feathered_kick=True,                # SIGNATURE - all four beats, soft
    walking_bass_style="swing",          # Classic swing bass
    freddie_green_guitar=True,          # SIGNATURE - 4-to-the-bar

    # ============== ARTICULATIONS ==============
    articulation_variety=0.4,            # Less variety than Ellington
    fall_probability=0.3,                # Some falls
    shake_probability=0.1,              # Minimal
    staccato_probability=0.7,           # PUNCHY, CRISP
    preferred_articulations=["staccato", "accent", "normal"],

    # ============== DYNAMICS ==============
    dynamic_range="medium",              # Not as extreme as Ellington
    use_crescendo=0.4,                  # Some builds
    shout_chorus_intensity=1.0,         # FAMOUS SHOUT CHORUSES

    # ============== FORM ==============
    intro_style="vamp",                  # Simple vamp or button
    ending_style="button",              # SHORT PUNCHY ENDING
    use_modulation=False,                # Typically stays in one key

    # ============== TEXTURE ==============
    texture_density=0.5,                 # Sparser than Ellington
    riff_based=True,                    # SIGNATURE - riff arrangements
    call_response=0.6,                   # Some call-response

    # ============== SWING FEEL ==============
    swing_ratio=0.62,                    # Medium swing
    swing_intensity=1.0,                 # Consistent, solid swing
    laid_back_feel=0.0,                  # Neutral, right in the pocket
)
