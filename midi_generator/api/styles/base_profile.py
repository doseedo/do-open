"""
Base Style Profile for Big Band Arranging

Defines the StyleProfile dataclass that encapsulates all arranging parameters
for a particular big band style/composer.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional


@dataclass
class StyleProfile:
    """
    Style profile for big band arranging

    This profile defines all the characteristics of a particular arranging style,
    from orchestration choices to harmonic complexity to dynamic range.

    Used by BigBandGenerator to configure the arrangement pipeline according to
    the chosen composer/style (Basie, Ellington, Thad Jones, etc.)
    """

    # ============== ORCHESTRATION ==============

    voicing_preference: str = "close"
    """Preferred voicing type: close, drop_2, drop_3, spread, open"""

    voicing_spacing: str = "consistent"
    """Voice spacing: consistent, varied, wide"""

    use_section_hits: float = 0.5
    """Probability of using section hits/stabs (0.0-1.0)"""

    use_riffs: float = 0.5
    """Probability of using background riffs (0.0-1.0)"""

    use_plunger_mutes: float = 0.0
    """Probability of plunger mute effects (0.0-1.0) - Ellington signature"""

    use_growls: float = 0.0
    """Probability of growl articulations (0.0-1.0)"""

    unusual_doublings: bool = False
    """Use unusual instrument doublings (Ellington: clarinet + muted trombone)"""

    # ============== HARMONY ==============

    harmony_complexity: float = 0.5
    """Overall harmonic complexity (0.0=simple, 1.0=very complex)"""

    chord_extensions: List[int] = field(default_factory=lambda: [7])
    """Chord extensions to use: [7] = basic, [7,9,11,13] = rich"""

    use_blues: float = 0.5
    """Emphasis on blues harmony (0.0-1.0)"""

    use_whole_tone: float = 0.0
    """Use whole tone scales (0.0-1.0) - Ellington"""

    use_diminished: float = 0.1
    """Use diminished scales (0.0-1.0)"""

    use_bitonal: float = 0.0
    """Use bitonal harmony (0.0-1.0) - advanced/Ellington"""

    use_modal_interchange: float = 0.2
    """Use modal interchange (0.0-1.0)"""

    use_tritone_subs: float = 0.3
    """Use tritone substitutions (0.0-1.0)"""

    reharmonization_level: float = 0.3
    """Overall reharmonization complexity (0.0=basic, 1.0=Bird-level)"""

    # ============== PIANO ==============

    piano_style: str = "comping"
    """Piano style: comping, stride, sparse, dense, block"""

    piano_density: float = 0.5
    """Piano note density (0.0=very sparse, 1.0=very dense)"""

    use_stride: bool = False
    """Use stride piano patterns"""

    rootless_voicings: bool = False
    """Use rootless piano voicings (Bill Evans style)"""

    # ============== RHYTHM SECTION ==============

    emphasis_on_rhythm: float = 0.5
    """How much emphasis on rhythm section (0.0-1.0)"""

    feathered_kick: bool = False
    """Use feathered kick drum (all four beats, soft) - Basie"""

    walking_bass_style: str = "bebop"
    """Walking bass style: bebop, swing, modern"""

    freddie_green_guitar: bool = False
    """Use Freddie Green 4-to-the-bar guitar - Basie"""

    # ============== ARTICULATIONS ==============

    articulation_variety: float = 0.5
    """Variety of articulations (0.0=minimal, 1.0=very varied)"""

    fall_probability: float = 0.3
    """Probability of fall articulations (0.0-1.0)"""

    shake_probability: float = 0.2
    """Probability of shake articulations (0.0-1.0)"""

    staccato_probability: float = 0.5
    """Probability of staccato notes (0.0-1.0)"""

    preferred_articulations: List[str] = field(
        default_factory=lambda: ["normal", "accent", "staccato"]
    )
    """List of preferred articulation types for this style"""

    # ============== DYNAMICS ==============

    dynamic_range: str = "medium"
    """Dynamic range: narrow, medium, wide, very_wide"""

    use_crescendo: float = 0.5
    """Use crescendo/diminuendo (0.0-1.0)"""

    shout_chorus_intensity: float = 0.8
    """Intensity of shout chorus (0.0-1.0)"""

    # ============== FORM ==============

    intro_style: str = "vamp"
    """Intro style: vamp, last_4, button, rubato, none"""

    ending_style: str = "tag"
    """Ending style: tag, fermata, ritardando, button"""

    use_modulation: bool = False
    """Use key modulation (typically up half-step for final chorus)"""

    # ============== TEXTURE ==============

    texture_density: float = 0.5
    """Overall texture density (0.0=sparse, 1.0=very dense)"""

    riff_based: bool = False
    """Arrangement is primarily riff-based (Basie)"""

    call_response: float = 0.5
    """Use call-and-response between sections (0.0-1.0)"""

    # ============== SWING FEEL ==============

    swing_ratio: float = 0.62
    """Swing ratio (0.5=straight, 0.67=triplet, 0.62=standard)"""

    swing_intensity: float = 1.0
    """Swing intensity/consistency (0.0-1.0)"""

    laid_back_feel: float = 0.0
    """Laid-back feel: notes slightly late (-1.0 to 1.0, 0=neutral)"""

    # ============== METADATA ==============

    style_name: str = "Generic"
    """Name of this style"""

    description: str = ""
    """Description of this arranging style"""

    composer_reference: str = ""
    """Reference composer/arranger"""

    era: str = "swing"
    """Musical era: swing, bebop, modern, contemporary"""

    typical_tempo_range: tuple = (120, 160)
    """Typical tempo range for this style (min_bpm, max_bpm)"""


    def get_suggested_tempo(self, requested_tempo: Optional[int] = None) -> int:
        """
        Get suggested tempo, adjusting if requested tempo is outside typical range.

        Args:
            requested_tempo: User-requested tempo or None

        Returns:
            Appropriate tempo for this style
        """
        if requested_tempo is None:
            # Return middle of typical range
            return (self.typical_tempo_range[0] + self.typical_tempo_range[1]) // 2

        min_tempo, max_tempo = self.typical_tempo_range

        # Warn if outside typical range but allow it
        if requested_tempo < min_tempo or requested_tempo > max_tempo:
            import warnings
            warnings.warn(
                f"{self.style_name} typically played at {min_tempo}-{max_tempo} BPM, "
                f"but {requested_tempo} BPM requested. Proceeding anyway."
            )

        return requested_tempo


    def __repr__(self) -> str:
        return (
            f"StyleProfile(name={self.style_name}, era={self.era}, "
            f"harmony_complexity={self.harmony_complexity}, "
            f"piano_style={self.piano_style})"
        )
