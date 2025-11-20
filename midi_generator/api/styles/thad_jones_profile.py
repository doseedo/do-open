"""
Thad Jones / Modern Big Band Style Profile

Based on Thad Jones and modern big band arrangers (Maria Schneider, Gordon Goodwin):
- Modern harmony (quartal voicings, clusters)
- Wide interval voicings (not close like swing era)
- Angular, contemporary melodies
- Rhythmic complexity
- Sophisticated voice leading
- Occasional odd meters
- Lush when needed, sparse when needed
- Contemporary jazz harmony (altered scales, symmetrical scales)

Reference recordings:
- Thad Jones: "A Child is Born", "Three and One"
- Maria Schneider: "Concert in the Garden"
- Gordon Goodwin: "Hunting Wabbits"
"""

from .base_profile import StyleProfile


THAD_JONES_STYLE = StyleProfile(
    # ============== METADATA ==============
    style_name="Thad Jones / Modern",
    description="Modern big band with quartal harmony and sophisticated voice leading",
    composer_reference="Thad Jones, Maria Schneider, Gordon Goodwin",
    era="modern",
    typical_tempo_range=(60, 200),  # Very wide - ballads to burners

    # ============== ORCHESTRATION ==============
    voicing_preference="quartal_and_clusters",  # MODERN VOICINGS
    voicing_spacing="wide_intervals",    # WIDE, OPEN - not close
    use_section_hits=0.6,                # Moderate
    use_riffs=0.4,                       # Less traditional riffs
    use_plunger_mutes=0.2,              # Occasional, not signature
    use_growls=0.1,                     # Minimal
    unusual_doublings=True,              # Modern orchestral colors

    # ============== HARMONY ==============
    harmony_complexity=0.8,              # VERY COMPLEX
    chord_extensions=[9, 11, 13, "altered"],  # FULL EXTENSIONS
    use_blues=0.4,                      # Some blues, but diverse
    use_whole_tone=0.3,                 # Modern harmonic palette
    use_diminished=0.4,                 # Symmetrical scales
    use_bitonal=0.3,                    # Modern technique
    use_modal_interchange=0.6,          # FREQUENT
    use_tritone_subs=0.6,               # Frequent reharmonization
    reharmonization_level=0.75,         # High sophistication

    # ============== PIANO ==============
    piano_style="comping",               # Modern comping
    piano_density=0.5,                  # Balanced
    use_stride=False,                   # Not modern style
    rootless_voicings=True,             # Bill Evans-style voicings

    # ============== RHYTHM SECTION ==============
    emphasis_on_rhythm=0.7,              # Important but balanced
    feathered_kick=False,                # Not modern style
    walking_bass_style="modern",         # Contemporary bass lines
    freddie_green_guitar=False,          # Not modern sound

    # ============== ARTICULATIONS ==============
    articulation_variety=0.7,            # HIGH variety
    fall_probability=0.4,                # Moderate
    shake_probability=0.2,              # Moderate
    staccato_probability=0.5,            # Mix
    preferred_articulations=["normal", "accent", "legato", "staccato"],

    # ============== DYNAMICS ==============
    dynamic_range="very_wide",           # PPP to FFF - full dynamic palette
    use_crescendo=0.8,                  # FREQUENT dynamic shaping
    shout_chorus_intensity=0.9,         # Strong climaxes

    # ============== FORM ==============
    intro_style="vamp",                  # Modern intros - vamps or rubato
    ending_style="tag",                 # Varied endings
    use_modulation=True,                # Frequent modulation

    # ============== TEXTURE ==============
    texture_density=0.7,                 # Varied - sparse to dense
    riff_based=False,                   # Through-composed modern writing
    call_response=0.6,                   # Moderate use

    # ============== SWING FEEL ==============
    swing_ratio=0.60,                    # LIGHTER SWING - more modern
    swing_intensity=0.85,                # Slightly looser, more flexible
    laid_back_feel=0.0,                  # Neutral
)
