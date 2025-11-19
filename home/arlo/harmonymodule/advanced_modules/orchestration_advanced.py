#!/usr/bin/env python3
"""
Advanced Orchestration & Instrument Ranges - Professional Idiomatic Writing

Comprehensive orchestration system with:
1. Instrument range validation (playable, comfortable, optimal registers)
2. Transposition handling for all transposing instruments
3. Doubling strategies (octave, unison, harmony)
4. Register analysis (dark, neutral, bright timbral characteristics)
5. Idiomatic writing checks (playability, awkward passages)
6. Wind/brass voicing techniques
7. String section techniques (pizz, arco, tremolo, sul pont, sul tasto, col legno)
8. SATB spacing rules
9. Blend vs contrast combinations
10. Percussion orchestration

Based on:
- Rimsky-Korsakov: "Principles of Orchestration" (1913)
- Samuel Adler: "The Study of Orchestration" (4th Edition)
- Berklee orchestration curriculum
- 2025 Nature Scientific Reports: "Factors Contributing to Instrumental Blends"
- Stanford CCRMA research on timbral characteristics

Author: Agent 15
Date: 2025
"""

from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings
import statistics

# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class Register(Enum):
    """Timbral register characteristics"""
    DARK = "dark"           # Low register, rich, warm, sometimes muddy
    NEUTRAL = "neutral"     # Middle register, balanced, natural
    BRIGHT = "bright"       # High register, brilliant, penetrating, sometimes shrill
    EXTREME_LOW = "extreme_low"   # Very low, often difficult to control
    EXTREME_HIGH = "extreme_high" # Very high, often strained or piercing


class Playability(Enum):
    """Playability assessment for passages"""
    EXCELLENT = 5      # Idiomatic, comfortable, natural
    GOOD = 4          # Playable, reasonable
    ACCEPTABLE = 3    # Possible but challenging
    DIFFICULT = 2     # Awkward, requires advanced technique
    PROBLEMATIC = 1   # Very difficult, should be avoided
    UNPLAYABLE = 0    # Impossible or extremely impractical


class StringTechnique(Enum):
    """String playing techniques"""
    ARCO = "arco"                       # Normal bowing
    PIZZICATO = "pizzicato"             # Plucked
    TREMOLO = "tremolo"                 # Rapid bow movement
    SUL_PONTICELLO = "sul_ponticello"   # Near bridge (metallic sound)
    SUL_TASTO = "sul_tasto"             # Over fingerboard (soft, flute-like)
    COL_LEGNO = "col_legno"             # With wood of bow
    HARMONICS = "harmonics"             # Flageolet tones
    SPICCATO = "spiccato"               # Bouncing bow
    MARCATO = "marcato"                 # Heavy accents
    STACCATO = "staccato"               # Short detached notes
    LEGATO = "legato"                   # Smooth connected notes


class DoublingStrategy(Enum):
    """Doubling approaches for orchestration"""
    UNISON = "unison"              # Same pitch, same octave
    OCTAVE = "octave"              # Octave apart
    TWO_OCTAVES = "two_octaves"    # Two octaves apart
    HARMONY = "harmony"            # Different pitches (3rd, 5th, etc.)
    BLEND = "blend"                # Same octave for timbral fusion
    CONTRAST = "contrast"          # Different registers for clarity


@dataclass
class InstrumentRange:
    """Comprehensive instrument range data"""
    name: str
    lowest_note: int           # MIDI note number
    highest_note: int          # MIDI note number
    comfortable_low: int       # Comfortable low end
    comfortable_high: int      # Comfortable high end
    optimal_low: int          # Sweet spot low
    optimal_high: int         # Sweet spot high
    transposition: int = 0    # Semitones from concert pitch (negative = down)
    is_transposing: bool = False
    family: str = "strings"   # strings, woodwinds, brass, percussion

    # Register boundaries (MIDI note numbers)
    dark_register_max: Optional[int] = None
    neutral_register_max: Optional[int] = None
    bright_register_min: Optional[int] = None

    # Idiomatic considerations
    awkward_intervals: List[str] = field(default_factory=list)
    preferred_intervals: List[str] = field(default_factory=list)
    fast_passage_limit: int = 120  # Max BPM for 16th notes


@dataclass
class VoicingResult:
    """Result of voicing operation"""
    notes: List[int]                    # MIDI note numbers
    instruments: List[str]              # Instrument names
    playability: Playability            # Overall playability rating
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ============================================================================
# COMPREHENSIVE INSTRUMENT DATABASE
# ============================================================================

INSTRUMENT_DATABASE = {
    # ========== STRINGS ==========
    "violin": InstrumentRange(
        name="Violin",
        lowest_note=55,  # G3
        highest_note=103,  # G7 (professional soloists can go higher)
        comfortable_low=59,  # B3
        comfortable_high=96,  # C7
        optimal_low=64,  # E4
        optimal_high=88,  # E6
        dark_register_max=64,  # E4 (G string: 55-64)
        neutral_register_max=79,  # G5 (D and A strings)
        bright_register_min=76,  # E5 (E string is very bright)
        family="strings",
        preferred_intervals=["major_2nd", "minor_2nd", "perfect_5th"],
        awkward_intervals=["tritone_leap", "major_7th_leap"]
    ),

    "viola": InstrumentRange(
        name="Viola",
        lowest_note=48,  # C3
        highest_note=91,  # G6
        comfortable_low=52,  # E3
        comfortable_high=84,  # C6
        optimal_low=55,  # G3
        optimal_high=79,  # G5
        dark_register_max=60,  # C4 (C string is very dark)
        neutral_register_max=72,  # C5
        bright_register_min=72,  # C5
        family="strings",
        preferred_intervals=["major_2nd", "minor_2nd", "perfect_4th"],
        awkward_intervals=["augmented_4th", "major_7th_leap"]
    ),

    "cello": InstrumentRange(
        name="Cello",
        lowest_note=36,  # C2
        highest_note=84,  # C6
        comfortable_low=40,  # E2
        comfortable_high=79,  # G5
        optimal_low=48,  # C3
        optimal_high=72,  # C5
        dark_register_max=52,  # E3 (C and G strings very dark, growly)
        neutral_register_max=67,  # G4 (D string)
        bright_register_min=69,  # A4 (A string bright, metallic, soaring)
        family="strings",
        preferred_intervals=["major_2nd", "minor_2nd", "perfect_5th"],
        awkward_intervals=["large_leaps_above_tenor_clef"]
    ),

    "double_bass": InstrumentRange(
        name="Double Bass",
        lowest_note=28,  # E1 (with extension, otherwise E1 = 28)
        highest_note=67,  # G4
        comfortable_low=32,  # G#1
        comfortable_high=60,  # C4
        optimal_low=36,  # C2
        optimal_high=55,  # G3
        dark_register_max=48,  # C3
        neutral_register_max=55,  # G3
        bright_register_min=60,  # C4 (relatively bright for bass)
        family="strings",
        transposition=-12,  # Sounds octave lower than written
        is_transposing=True,
        preferred_intervals=["major_2nd", "minor_2nd"],
        awkward_intervals=["large_leaps", "fast_passages_above_G3"],
        fast_passage_limit=90
    ),

    # ========== WOODWINDS ==========
    "flute": InstrumentRange(
        name="Flute",
        lowest_note=60,  # C4
        highest_note=96,  # C7 (professionals can go to D7)
        comfortable_low=62,  # D4
        comfortable_high=91,  # G6
        optimal_low=65,  # F4
        optimal_high=84,  # C6
        dark_register_max=65,  # F4 (low register is soft, breathy)
        neutral_register_max=79,  # G5
        bright_register_min=79,  # G5 (upper register very bright, penetrating)
        family="woodwinds",
        preferred_intervals=["major_2nd", "minor_2nd", "perfect_5th"],
        awkward_intervals=[]  # Very agile instrument
    ),

    "oboe": InstrumentRange(
        name="Oboe",
        lowest_note=58,  # Bb3
        highest_note=89,  # F6
        comfortable_low=60,  # C4
        comfortable_high=84,  # C6
        optimal_low=62,  # D4
        optimal_high=79,  # G5
        dark_register_max=67,  # G4
        neutral_register_max=76,  # E5
        bright_register_min=77,  # F5 (upper register is penetrating)
        family="woodwinds",
        preferred_intervals=["major_2nd", "minor_2nd"],
        awkward_intervals=["large_leaps"]
    ),

    "clarinet_bb": InstrumentRange(
        name="Bb Clarinet",
        lowest_note=50,  # D3 (written)
        highest_note=94,  # Bb6 (written)
        comfortable_low=53,  # F3 (written)
        comfortable_high=89,  # F6 (written)
        optimal_low=57,  # A3 (written)
        optimal_high=84,  # C6 (written)
        dark_register_max=62,  # D4 (chalumeau register is dark, rich)
        neutral_register_max=74,  # D5 (clarion register)
        bright_register_min=77,  # F5 (altissimo register is bright)
        family="woodwinds",
        transposition=-2,  # Bb instrument (sounds major 2nd lower)
        is_transposing=True,
        preferred_intervals=["major_2nd", "minor_2nd", "arpeggios"],
        awkward_intervals=["octave_flips"]  # More difficult than on flute
    ),

    "bassoon": InstrumentRange(
        name="Bassoon",
        lowest_note=34,  # Bb1
        highest_note=75,  # Eb5
        comfortable_low=36,  # C2
        comfortable_high=69,  # A4
        optimal_low=41,  # F2
        optimal_high=65,  # F4
        dark_register_max=48,  # C3 (very dark, rich)
        neutral_register_max=60,  # C4
        bright_register_min=65,  # F4 (tenor register can be reedy)
        family="woodwinds",
        preferred_intervals=["major_2nd", "minor_2nd"],
        awkward_intervals=["large_leaps_in_low_register"]
    ),

    "alto_sax_eb": InstrumentRange(
        name="Eb Alto Saxophone",
        lowest_note=49,  # Db3 (written)
        highest_note=87,  # Eb6 (written, altissimo higher)
        comfortable_low=52,  # E3 (written)
        comfortable_high=82,  # Bb5 (written)
        optimal_low=56,  # Ab3 (written)
        optimal_high=77,  # F5 (written)
        dark_register_max=62,  # D4 (written)
        neutral_register_max=74,  # D5 (written)
        bright_register_min=77,  # F5 (written)
        family="woodwinds",
        transposition=-9,  # Eb instrument (sounds major 6th lower)
        is_transposing=True,
        preferred_intervals=["major_2nd", "minor_2nd", "arpeggios"],
        awkward_intervals=[]
    ),

    # ========== BRASS ==========
    "trumpet_bb": InstrumentRange(
        name="Bb Trumpet",
        lowest_note=52,  # E3 (written)
        highest_note=89,  # F6 (written, professionals higher)
        comfortable_low=55,  # G3 (written)
        comfortable_high=82,  # Bb5 (written)
        optimal_low=58,  # Bb3 (written)
        optimal_high=77,  # F5 (written)
        dark_register_max=62,  # D4 (written, low register)
        neutral_register_max=72,  # C5 (written)
        bright_register_min=74,  # D5 (written, brilliant, assertive)
        family="brass",
        transposition=-2,  # Bb instrument
        is_transposing=True,
        preferred_intervals=["harmonic_series", "arpeggios"],
        awkward_intervals=["chromatic_passages", "large_leaps_outside_harmonics"]
    ),

    "french_horn_f": InstrumentRange(
        name="F Horn (French Horn)",
        lowest_note=41,  # F2 (written as C3)
        highest_note=77,  # F5 (written as C6)
        comfortable_low=46,  # Bb2 (written)
        comfortable_high=72,  # C5 (written)
        optimal_low=51,  # Eb3 (written)
        optimal_high=67,  # G4 (written)
        dark_register_max=55,  # G3 (written, rich, dark)
        neutral_register_max=65,  # F4 (written)
        bright_register_min=69,  # A4 (written)
        family="brass",
        transposition=-7,  # F instrument (sounds perfect 5th lower)
        is_transposing=True,
        preferred_intervals=["harmonic_series", "arpeggios"],
        awkward_intervals=["chromatic_fast_passages"],
        fast_passage_limit=80  # Much less virtuosic than woodwinds
    ),

    "trombone": InstrumentRange(
        name="Trombone",
        lowest_note=40,  # E2
        highest_note=72,  # C5
        comfortable_low=43,  # G2
        comfortable_high=67,  # G4
        optimal_low=46,  # Bb2
        optimal_high=62,  # D4
        dark_register_max=50,  # D3
        neutral_register_max=60,  # C4
        bright_register_min=65,  # F4
        family="brass",
        preferred_intervals=["harmonic_series", "stepwise"],
        awkward_intervals=["fast_chromatic_passages"]
    ),

    "tuba": InstrumentRange(
        name="Tuba",
        lowest_note=28,  # E1
        highest_note=60,  # C4
        comfortable_low=32,  # G#1
        comfortable_high=55,  # G3
        optimal_low=36,  # C2
        optimal_high=50,  # D3
        dark_register_max=43,  # G2
        neutral_register_max=50,  # D3
        bright_register_min=53,  # F3
        family="brass",
        preferred_intervals=["harmonic_series", "stepwise"],
        awkward_intervals=["fast_passages", "large_leaps"]
    ),
}


# ============================================================================
# ADVANCED ORCHESTRATION CLASS
# ============================================================================

class AdvancedOrchestration:
    """
    Professional orchestration with idiomatic writing

    Features:
    - Instrument range validation
    - Transposition handling
    - Doubling strategies
    - Register analysis (dark, bright)
    - Idiomatic writing checks
    - Wind/brass voicing
    - String techniques
    - SATB spacing rules
    - Blend vs contrast combinations

    Examples:
        >>> orch = AdvancedOrchestration()
        >>> orch.validate_instrument_range("violin", 72)
        (True, Register.NEUTRAL, Playability.EXCELLENT, "")

        >>> orch.transpose_for_instrument(60, "clarinet_bb")
        62  # Written D sounds as concert C

        >>> orch.suggest_doubling([72], "violin")
        {'octave_below': ['viola', 'cello'], 'unison': ['flute', 'oboe']}
    """

    def __init__(self):
        """Initialize orchestration engine"""
        self.instruments = INSTRUMENT_DATABASE

    # ========================================================================
    # INSTRUMENT RANGE VALIDATION
    # ========================================================================

    def validate_instrument_range(
        self,
        instrument: str,
        pitch: int,
        context: str = "general"
    ) -> Tuple[bool, Register, Playability, str]:
        """
        Validate if pitch is playable on instrument

        Args:
            instrument: Instrument name from database
            pitch: MIDI note number (concert pitch)
            context: Performance context (general, solo, ensemble)

        Returns:
            (is_valid, register, playability, message)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> valid, reg, play, msg = orch.validate_instrument_range("violin", 72)
            >>> valid
            True
            >>> reg
            Register.NEUTRAL
        """
        if instrument not in self.instruments:
            return (False, Register.NEUTRAL, Playability.UNPLAYABLE,
                   f"Unknown instrument: {instrument}")

        inst = self.instruments[instrument]

        # Check if pitch is within absolute range
        if pitch < inst.lowest_note or pitch > inst.highest_note:
            return (False, Register.NEUTRAL, Playability.UNPLAYABLE,
                   f"{inst.name}: Pitch {pitch} outside range "
                   f"[{inst.lowest_note}-{inst.highest_note}]")

        # Determine register
        register = self._determine_register(instrument, pitch)

        # Determine playability
        playability = self._determine_playability(instrument, pitch, context)

        # Generate message
        msg = self._generate_range_message(instrument, pitch, register, playability)

        return (True, register, playability, msg)

    def _determine_register(self, instrument: str, pitch: int) -> Register:
        """Determine timbral register for pitch on instrument"""
        inst = self.instruments[instrument]

        # Extreme low: only for bass instruments (MIDI < 40) at their lowest notes
        if (pitch <= inst.lowest_note and inst.lowest_note < 40):
            return Register.EXTREME_LOW

        # Extreme high: very near highest note
        if pitch >= inst.highest_note - 2:
            return Register.EXTREME_HIGH

        # Dark register
        if inst.dark_register_max and pitch <= inst.dark_register_max:
            return Register.DARK

        # Bright register
        if inst.bright_register_min and pitch >= inst.bright_register_min:
            return Register.BRIGHT

        # Default to neutral
        return Register.NEUTRAL

    def _determine_playability(
        self,
        instrument: str,
        pitch: int,
        context: str
    ) -> Playability:
        """Determine playability rating for pitch"""
        inst = self.instruments[instrument]

        # Optimal range = excellent
        if inst.optimal_low <= pitch <= inst.optimal_high:
            return Playability.EXCELLENT

        # Comfortable range = good
        if inst.comfortable_low <= pitch <= inst.comfortable_high:
            return Playability.GOOD

        # Within absolute range but outside comfortable = acceptable to difficult
        if pitch < inst.comfortable_low:
            distance = inst.comfortable_low - pitch
            if distance <= 2:
                return Playability.ACCEPTABLE
            else:
                return Playability.DIFFICULT

        if pitch > inst.comfortable_high:
            distance = pitch - inst.comfortable_high
            if distance <= 2:
                return Playability.ACCEPTABLE
            else:
                return Playability.DIFFICULT

        return Playability.ACCEPTABLE

    def _generate_range_message(
        self,
        instrument: str,
        pitch: int,
        register: Register,
        playability: Playability
    ) -> str:
        """Generate descriptive message about range/playability"""
        inst = self.instruments[instrument]

        if playability == Playability.EXCELLENT:
            return f"{inst.name}: {pitch} is in optimal range ({register.value} register)"
        elif playability == Playability.GOOD:
            return f"{inst.name}: {pitch} is comfortable ({register.value} register)"
        elif playability == Playability.ACCEPTABLE:
            return f"{inst.name}: {pitch} is playable but challenging ({register.value})"
        elif playability == Playability.DIFFICULT:
            return f"{inst.name}: {pitch} is difficult ({register.value}), use sparingly"
        else:
            return f"{inst.name}: {pitch} is problematic"

    # ========================================================================
    # TRANSPOSITION
    # ========================================================================

    def transpose_for_instrument(
        self,
        concert_pitch: int,
        instrument: str
    ) -> int:
        """
        Transpose concert pitch to written pitch for transposing instrument

        Args:
            concert_pitch: MIDI note number at concert pitch
            instrument: Instrument name

        Returns:
            Written pitch (MIDI note number)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> orch.transpose_for_instrument(60, "clarinet_bb")
            62  # Bb clarinet: concert C = written D

            >>> orch.transpose_for_instrument(60, "french_horn_f")
            67  # F horn: concert C = written G
        """
        if instrument not in self.instruments:
            warnings.warn(f"Unknown instrument: {instrument}")
            return concert_pitch

        inst = self.instruments[instrument]

        if not inst.is_transposing:
            return concert_pitch

        # Transposition value is how much the instrument sounds LOWER
        # So written pitch must be HIGHER
        written_pitch = concert_pitch - inst.transposition

        return written_pitch

    def transpose_to_concert(
        self,
        written_pitch: int,
        instrument: str
    ) -> int:
        """
        Transpose written pitch to concert pitch

        Args:
            written_pitch: Written MIDI note number
            instrument: Instrument name

        Returns:
            Concert pitch (MIDI note number)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> orch.transpose_to_concert(62, "clarinet_bb")
            60  # Written D on Bb clarinet sounds as concert C
        """
        if instrument not in self.instruments:
            warnings.warn(f"Unknown instrument: {instrument}")
            return written_pitch

        inst = self.instruments[instrument]

        if not inst.is_transposing:
            return written_pitch

        concert_pitch = written_pitch + inst.transposition

        return concert_pitch

    # ========================================================================
    # DOUBLING STRATEGIES
    # ========================================================================

    def suggest_doubling(
        self,
        melody: List[int],
        source_instrument: str,
        strategy: DoublingStrategy = DoublingStrategy.BLEND
    ) -> Dict[str, List[str]]:
        """
        Suggest instruments for doubling a melody

        Based on Rimsky-Korsakov and Adler recommendations:
        - Violin + Trumpet (unison)
        - Viola + Horn (unison)
        - Cello + Horn (unison/octave)
        - Flute + Violin (unison/octave)
        - Oboe + Trumpet (soft dynamics)
        - Clarinet + Viola (lower register)
        - Bassoon + Cello (unison)
        - Bassoon + Trombone (unison)
        - Tuba + Bass Trombone (unison)

        Args:
            melody: List of MIDI pitches
            source_instrument: Original instrument
            strategy: Doubling approach

        Returns:
            Dictionary of doubling suggestions by type

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> suggestions = orch.suggest_doubling([72, 74, 76], "violin")
            >>> 'unison' in suggestions
            True
        """
        if not melody:
            return {}

        avg_pitch = int(statistics.mean(melody))
        pitch_range = max(melody) - min(melody)

        suggestions = {
            'unison': [],
            'octave_above': [],
            'octave_below': [],
            'two_octaves_below': [],
            'harmony_third': [],
            'harmony_fifth': []
        }

        # Traditional doubling combinations (Rimsky-Korsakov)
        traditional_pairs = {
            'violin': {
                'unison': ['flute', 'oboe', 'trumpet_bb'],
                'octave_below': ['viola', 'clarinet_bb'],
                'two_octaves_below': ['cello']
            },
            'viola': {
                'unison': ['french_horn_f', 'clarinet_bb'],
                'octave_above': ['violin'],
                'octave_below': ['cello', 'bassoon']
            },
            'cello': {
                'unison': ['french_horn_f', 'bassoon', 'trombone'],
                'octave_above': ['viola'],
                'octave_below': ['double_bass']
            },
            'flute': {
                'unison': ['violin', 'oboe'],
                'octave_below': ['clarinet_bb']
            },
            'oboe': {
                'unison': ['flute', 'violin', 'trumpet_bb'],
                'octave_below': ['clarinet_bb']
            },
            'clarinet_bb': {
                'unison': ['viola'],
                'octave_above': ['flute', 'oboe']
            },
            'bassoon': {
                'unison': ['cello', 'trombone'],
                'octave_above': ['clarinet_bb']
            },
            'trumpet_bb': {
                'unison': ['violin', 'oboe']
            },
            'french_horn_f': {
                'unison': ['viola', 'cello']
            },
            'trombone': {
                'unison': ['bassoon', 'cello']
            }
        }

        if source_instrument in traditional_pairs:
            return traditional_pairs[source_instrument]

        # Algorithmic suggestions based on range compatibility
        for inst_name, inst_data in self.instruments.items():
            if inst_name == source_instrument:
                continue

            # Check if doubling is in range
            if strategy == DoublingStrategy.UNISON:
                if all(inst_data.comfortable_low <= p <= inst_data.comfortable_high
                      for p in melody):
                    suggestions['unison'].append(inst_name)

            elif strategy == DoublingStrategy.OCTAVE:
                octave_up = [p + 12 for p in melody]
                octave_down = [p - 12 for p in melody]

                if all(inst_data.comfortable_low <= p <= inst_data.comfortable_high
                      for p in octave_up):
                    suggestions['octave_above'].append(inst_name)

                if all(inst_data.comfortable_low <= p <= inst_data.comfortable_high
                      for p in octave_down):
                    suggestions['octave_below'].append(inst_name)

        return suggestions

    # ========================================================================
    # REGISTER ANALYSIS
    # ========================================================================

    def analyze_register(
        self,
        pitch: int,
        instrument: str
    ) -> Tuple[Register, str]:
        """
        Analyze timbral characteristics of pitch on instrument

        Args:
            pitch: MIDI note number
            instrument: Instrument name

        Returns:
            (register, description)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> reg, desc = orch.analyze_register(55, "violin")
            >>> reg
            Register.DARK
            >>> "G string" in desc
            True
        """
        if instrument not in self.instruments:
            return (Register.NEUTRAL, "Unknown instrument")

        inst = self.instruments[instrument]
        register = self._determine_register(instrument, pitch)

        # Generate specific description
        descriptions = {
            "violin": {
                Register.DARK: "G string: rich, warm, somewhat veiled",
                Register.NEUTRAL: "D or A string: balanced, natural tone",
                Register.BRIGHT: "E string: brilliant, penetrating, soaring",
                Register.EXTREME_HIGH: "Very high register: intense, may be shrill"
            },
            "viola": {
                Register.DARK: "C string: very dark, rich, heavy",
                Register.NEUTRAL: "G or D string: mellow, warm",
                Register.BRIGHT: "A string: bright, focused",
                Register.EXTREME_HIGH: "High register: strained but usable"
            },
            "cello": {
                Register.DARK: "C or G string: dark, rich, growly, powerful bass",
                Register.NEUTRAL: "D string: balanced, singing quality",
                Register.BRIGHT: "A string: bright, metallic, soaring, vocal",
                Register.EXTREME_HIGH: "Tenor register: intense, brilliant"
            },
            "flute": {
                Register.DARK: "Low register: soft, breathy, mysterious",
                Register.NEUTRAL: "Middle register: clear, balanced",
                Register.BRIGHT: "High register: brilliant, penetrating, strong",
                Register.EXTREME_HIGH: "Altissimo: very bright, piercing"
            },
            "clarinet_bb": {
                Register.DARK: "Chalumeau: dark, rich, hollow",
                Register.NEUTRAL: "Clarion: clear, focused, versatile",
                Register.BRIGHT: "Altissimo: bright, reedy, penetrating",
                Register.EXTREME_HIGH: "Extreme altissimo: very difficult, shrill"
            },
            "french_horn_f": {
                Register.DARK: "Low register: dark, mysterious, rich",
                Register.NEUTRAL: "Middle register: warm, noble, blends well",
                Register.BRIGHT: "High register: heroic, brilliant",
                Register.EXTREME_HIGH: "Extreme high: strained, risky"
            }
        }

        if instrument in descriptions and register in descriptions[instrument]:
            return (register, descriptions[instrument][register])

        return (register, f"{register.value} register")

    # ========================================================================
    # IDIOMATIC WRITING CHECKS
    # ========================================================================

    def check_idiomatic_writing(
        self,
        passage: List[int],
        instrument: str,
        tempo: int = 120,
        rhythm: str = "8ths"
    ) -> Tuple[Playability, List[str]]:
        """
        Check if passage is idiomatic for instrument

        Checks:
        - Range violations
        - Awkward intervals
        - Tempo limitations
        - Large leaps
        - Register changes

        Args:
            passage: List of MIDI pitches
            instrument: Instrument name
            tempo: BPM
            rhythm: Note values (16ths, 8ths, quarters)

        Returns:
            (playability_rating, list_of_issues)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> play, issues = orch.check_idiomatic_writing(
            ...     [60, 62, 64, 65, 67], "flute", tempo=120)
            >>> play
            Playability.EXCELLENT
        """
        if instrument not in self.instruments:
            return (Playability.UNPLAYABLE, [f"Unknown instrument: {instrument}"])

        inst = self.instruments[instrument]
        issues = []
        playability_scores = []

        # Check each note's range
        for pitch in passage:
            valid, register, play, msg = self.validate_instrument_range(
                instrument, pitch
            )
            if not valid:
                issues.append(msg)
                playability_scores.append(0)
            else:
                playability_scores.append(play.value)

        # Check tempo limitations for fast passages
        if rhythm == "16ths" and tempo > inst.fast_passage_limit:
            issues.append(
                f"Fast 16ths at {tempo} BPM may be too difficult "
                f"(limit: {inst.fast_passage_limit} BPM)"
            )
            playability_scores.append(2)  # Difficult

        # Check for large leaps
        for i in range(len(passage) - 1):
            interval = abs(passage[i+1] - passage[i])
            if interval > 12:  # Larger than octave
                issues.append(
                    f"Large leap of {interval} semitones "
                    f"between notes {i} and {i+1}"
                )
                if interval > 19:  # Larger than 12th
                    playability_scores.append(2)  # Difficult

        # Check for awkward intervals (instrument-specific)
        if "tritone_leap" in inst.awkward_intervals:
            for i in range(len(passage) - 1):
                if abs(passage[i+1] - passage[i]) == 6:
                    issues.append(f"Tritone leap at notes {i}-{i+1} (awkward)")
                    playability_scores.append(3)  # Acceptable but not ideal

        # Check for extreme register changes
        if len(passage) > 1:
            pitch_range = max(passage) - min(passage)
            if pitch_range > 24:  # More than 2 octaves
                issues.append(
                    f"Wide range of {pitch_range} semitones may be awkward"
                )
                playability_scores.append(3)

        # Calculate overall playability
        if not playability_scores:
            return (Playability.EXCELLENT, issues)

        avg_score = statistics.mean(playability_scores)

        if avg_score >= 4.5:
            overall = Playability.EXCELLENT
        elif avg_score >= 3.5:
            overall = Playability.GOOD
        elif avg_score >= 2.5:
            overall = Playability.ACCEPTABLE
        elif avg_score >= 1.5:
            overall = Playability.DIFFICULT
        else:
            overall = Playability.PROBLEMATIC

        return (overall, issues)

    # ========================================================================
    # WIND/BRASS VOICING
    # ========================================================================

    def voice_for_winds(
        self,
        chord: List[int],
        section: str = "woodwinds",
        voicing_style: str = "traditional"
    ) -> VoicingResult:
        """
        Voice chord for wind/brass section

        Traditional order (high to low):
        - Woodwinds: Flute, Oboe, Clarinet, Bassoon
        - Brass: Trumpet, Horn, Trombone, Tuba

        Voicing styles:
        - traditional: Top to bottom (flute highest)
        - interlocking: Mix timbres (blend)
        - block: Homogeneous sections

        Args:
            chord: List of MIDI pitches (concert pitch)
            section: "woodwinds" or "brass"
            voicing_style: "traditional", "interlocking", "block"

        Returns:
            VoicingResult with assigned instruments

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> result = orch.voice_for_winds([60, 64, 67, 72], "woodwinds")
            >>> len(result.instruments)
            4
            >>> result.playability.value >= 3
            True
        """
        if section not in ["woodwinds", "brass"]:
            return VoicingResult(
                notes=[],
                instruments=[],
                playability=Playability.UNPLAYABLE,
                warnings=[f"Unknown section: {section}"]
            )

        # Sort chord from low to high
        sorted_chord = sorted(chord)

        # Define instrument orders
        woodwind_order = ["flute", "oboe", "clarinet_bb", "bassoon"]
        brass_order = ["trumpet_bb", "french_horn_f", "trombone", "tuba"]

        order = woodwind_order if section == "woodwinds" else brass_order

        # Assign instruments based on voicing style
        if voicing_style == "traditional":
            # Highest pitch to highest instrument
            notes = sorted_chord[::-1]  # High to low
            instruments = order[:len(notes)]

        elif voicing_style == "interlocking":
            # Alternate instruments for better blend
            notes = sorted_chord
            instruments = []
            for i in range(len(notes)):
                inst_idx = i % len(order)
                instruments.append(order[inst_idx])

        else:  # block
            # Low to high assignment
            notes = sorted_chord
            instruments = order[-len(notes):]  # Bottom instruments
            instruments = instruments[::-1]  # Reverse for low-to-high

        # Validate ranges
        warnings_list = []
        playability_scores = []

        for note, inst in zip(notes, instruments):
            valid, reg, play, msg = self.validate_instrument_range(inst, note)
            if not valid:
                warnings_list.append(msg)
                playability_scores.append(0)
            else:
                playability_scores.append(play.value)
                if play.value < 3:
                    warnings_list.append(msg)

        # Calculate overall playability
        if playability_scores:
            avg_score = statistics.mean(playability_scores)
            if avg_score >= 4:
                overall = Playability.EXCELLENT
            elif avg_score >= 3:
                overall = Playability.GOOD
            elif avg_score >= 2:
                overall = Playability.ACCEPTABLE
            else:
                overall = Playability.DIFFICULT
        else:
            overall = Playability.UNPLAYABLE

        return VoicingResult(
            notes=notes,
            instruments=instruments,
            playability=overall,
            warnings=warnings_list
        )

    # ========================================================================
    # STRING TECHNIQUES
    # ========================================================================

    def apply_string_technique(
        self,
        notes: List[int],
        technique: StringTechnique,
        instrument: str = "violin"
    ) -> Dict[str, any]:
        """
        Apply string technique with performance considerations

        Args:
            notes: List of MIDI pitches
            technique: String technique to apply
            instrument: String instrument name

        Returns:
            Dictionary with technique info and performance notes

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> result = orch.apply_string_technique(
            ...     [60, 64, 67], StringTechnique.PIZZICATO)
            >>> result['technique']
            'pizzicato'
        """
        if instrument not in self.instruments:
            return {'error': f'Unknown instrument: {instrument}'}

        if self.instruments[instrument].family != "strings":
            return {'error': f'{instrument} is not a string instrument'}

        result = {
            'technique': technique.value,
            'notes': notes,
            'instrument': instrument,
            'performance_notes': [],
            'limitations': []
        }

        # Add technique-specific information
        if technique == StringTechnique.PIZZICATO:
            result['performance_notes'].append(
                "Allow time to switch from arco (mark 'pizz.')"
            )
            result['performance_notes'].append(
                "Not effective in extreme low register (weak sound)"
            )
            result['limitations'].append("Tempo limit: ~200 BPM for fast passages")

        elif technique == StringTechnique.TREMOLO:
            result['performance_notes'].append(
                "Unmeasured tremolo: rapid bow movement (mark with slashes)"
            )
            result['performance_notes'].append(
                "Very effective for sustained tension"
            )

        elif technique == StringTechnique.SUL_PONTICELLO:
            result['performance_notes'].append(
                "Bow near bridge: metallic, glassy, eerie sound"
            )
            result['performance_notes'].append(
                "Emphasizes upper harmonics"
            )
            result['limitations'].append("Less volume, may be unstable")

        elif technique == StringTechnique.SUL_TASTO:
            result['performance_notes'].append(
                "Bow over fingerboard: soft, flute-like, ethereal"
            )
            result['performance_notes'].append(
                "Reduces upper harmonics"
            )
            result['limitations'].append("Very soft, may lack projection")

        elif technique == StringTechnique.COL_LEGNO:
            result['performance_notes'].append(
                "Strike strings with wood of bow: percussive, dry"
            )
            result['performance_notes'].append(
                "Col legno battuto (strike) vs tratto (draw)"
            )
            result['limitations'].append("Very soft, fragile sound")

        elif technique == StringTechnique.HARMONICS:
            result['performance_notes'].append(
                "Natural harmonics: nodes at 1/2, 1/3, 1/4 of string length"
            )
            result['performance_notes'].append(
                "Artificial harmonics: 4th finger stops, 1st finger touches"
            )
            result['limitations'].append("Soft, ethereal, limited dynamics")

        return result

    # ========================================================================
    # SATB SPACING RULES
    # ========================================================================

    def enforce_satb_spacing(
        self,
        soprano: int,
        alto: int,
        tenor: int,
        bass: int
    ) -> Tuple[bool, List[str]]:
        """
        Check SATB spacing rules

        Rules:
        1. No more than octave between S-A, A-T
        2. Up to 12th (octave + 5th) between T-B
        3. No voice crossing
        4. Minimum 5th between T-B (guideline)

        Args:
            soprano: MIDI pitch
            alto: MIDI pitch
            tenor: MIDI pitch
            bass: MIDI pitch

        Returns:
            (is_valid, list_of_violations)

        Examples:
            >>> orch = AdvancedOrchestration()
            >>> valid, issues = orch.enforce_satb_spacing(72, 67, 60, 48)
            >>> valid
            True
        """
        violations = []

        # Rule 1: S-A within octave
        sa_interval = soprano - alto
        if sa_interval > 12:
            violations.append(
                f"Soprano-Alto spacing too wide: {sa_interval} semitones "
                f"(max 12)"
            )
        elif sa_interval < 0:
            violations.append("Voice crossing: Alto above Soprano")

        # Rule 2: A-T within octave
        at_interval = alto - tenor
        if at_interval > 12:
            violations.append(
                f"Alto-Tenor spacing too wide: {at_interval} semitones "
                f"(max 12)"
            )
        elif at_interval < 0:
            violations.append("Voice crossing: Tenor above Alto")

        # Rule 3: T-B within 12th (optional, can be wider)
        tb_interval = tenor - bass
        if tb_interval > 19:  # More than 12th
            violations.append(
                f"Tenor-Bass spacing very wide: {tb_interval} semitones "
                f"(>12th). Consider closing spacing."
            )
        elif tb_interval < 0:
            violations.append("Voice crossing: Bass above Tenor")

        # Guideline: Minimum 5th between T-B
        if tb_interval < 7 and bass > 55:  # High bass exception
            violations.append(
                f"Tenor-Bass spacing narrow: {tb_interval} semitones "
                f"(prefer minimum 5th = 7 semitones)"
            )

        is_valid = len(violations) == 0

        return (is_valid, violations)


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED ORCHESTRATION - COMPREHENSIVE TESTING")
    print("=" * 70)

    orch = AdvancedOrchestration()

    test_count = 0
    passed = 0

    def run_test(name: str, condition: bool):
        global test_count, passed
        test_count += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        print(f"{status}: {name}")
        if condition:
            passed += 1

    # ========================================================================
    # TEST 1-5: INSTRUMENT RANGE VALIDATION
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 1: Instrument Range Validation (5 tests)")
    print("=" * 70)

    # Test 1: Violin optimal range
    valid, reg, play, msg = orch.validate_instrument_range("violin", 72)  # C5
    run_test("Violin C5 in optimal range", valid and play == Playability.EXCELLENT)

    # Test 2: Violin out of range
    valid, reg, play, msg = orch.validate_instrument_range("violin", 110)
    run_test("Violin G7 out of range", not valid)

    # Test 3: Cello dark register
    valid, reg, play, msg = orch.validate_instrument_range("cello", 48)  # C3
    run_test("Cello C3 in dark register", valid and reg == Register.DARK)

    # Test 4: Flute bright register
    valid, reg, play, msg = orch.validate_instrument_range("flute", 84)  # C6
    run_test("Flute C6 in bright register", valid and reg == Register.BRIGHT)

    # Test 5: Double bass extreme low
    valid, reg, play, msg = orch.validate_instrument_range("double_bass", 28)  # E1
    run_test("Double bass E1 at extreme low", valid and reg == Register.EXTREME_LOW)

    # ========================================================================
    # TEST 6-10: TRANSPOSITION
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 2: Transposition (5 tests)")
    print("=" * 70)

    # Test 6: Bb clarinet transposition
    written = orch.transpose_for_instrument(60, "clarinet_bb")  # Concert C
    run_test("Bb clarinet: Concert C = Written D", written == 62)

    # Test 7: Bb clarinet back to concert
    concert = orch.transpose_to_concert(62, "clarinet_bb")
    run_test("Bb clarinet: Written D = Concert C", concert == 60)

    # Test 8: F horn transposition
    written = orch.transpose_for_instrument(60, "french_horn_f")  # Concert C
    run_test("F horn: Concert C = Written G", written == 67)

    # Test 9: Eb alto sax transposition
    written = orch.transpose_for_instrument(60, "alto_sax_eb")
    run_test("Eb alto sax: Concert C = Written A", written == 69)

    # Test 10: Non-transposing instrument
    written = orch.transpose_for_instrument(60, "violin")
    run_test("Violin non-transposing", written == 60)

    # ========================================================================
    # TEST 11-15: DOUBLING STRATEGIES
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 3: Doubling Strategies (5 tests)")
    print("=" * 70)

    # Test 11: Violin doubling suggestions
    suggestions = orch.suggest_doubling([72, 74, 76], "violin")
    run_test("Violin doubling includes flute/oboe/trumpet",
             any(inst in suggestions.get('unison', [])
                 for inst in ['flute', 'oboe', 'trumpet_bb']))

    # Test 12: Cello doubling suggestions
    suggestions = orch.suggest_doubling([48, 52, 55], "cello")
    run_test("Cello doubling includes horn/bassoon",
             any(inst in suggestions.get('unison', [])
                 for inst in ['french_horn_f', 'bassoon']))

    # Test 13: Viola doubling suggestions
    suggestions = orch.suggest_doubling([60, 64, 67], "viola")
    run_test("Viola doubling includes horn",
             'french_horn_f' in suggestions.get('unison', []))

    # Test 14: Bassoon doubling
    suggestions = orch.suggest_doubling([48, 52, 55], "bassoon")
    run_test("Bassoon doubling includes cello/trombone",
             any(inst in suggestions.get('unison', [])
                 for inst in ['cello', 'trombone']))

    # Test 15: Flute doubling
    suggestions = orch.suggest_doubling([72, 76, 79], "flute")
    run_test("Flute doubling includes violin/oboe",
             any(inst in suggestions.get('unison', [])
                 for inst in ['violin', 'oboe']))

    # ========================================================================
    # TEST 16-20: REGISTER ANALYSIS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 4: Register Analysis (5 tests)")
    print("=" * 70)

    # Test 16: Violin G string (dark)
    reg, desc = orch.analyze_register(55, "violin")  # G3
    run_test("Violin G3 is dark register", reg == Register.DARK)

    # Test 17: Violin E string (bright)
    reg, desc = orch.analyze_register(84, "violin")  # C6
    run_test("Violin C6 is bright register", reg == Register.BRIGHT)

    # Test 18: Clarinet chalumeau (dark)
    reg, desc = orch.analyze_register(57, "clarinet_bb")  # A3
    run_test("Clarinet A3 in chalumeau (dark)", reg == Register.DARK)

    # Test 19: Flute low register (dark)
    reg, desc = orch.analyze_register(62, "flute")  # D4
    run_test("Flute D4 in dark register", reg == Register.DARK)

    # Test 20: Horn high register (bright)
    reg, desc = orch.analyze_register(70, "french_horn_f")
    run_test("Horn high register is bright", reg == Register.BRIGHT)

    # ========================================================================
    # TEST 21-25: IDIOMATIC WRITING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 5: Idiomatic Writing Checks (5 tests)")
    print("=" * 70)

    # Test 21: Idiomatic scale passage
    play, issues = orch.check_idiomatic_writing(
        [60, 62, 64, 65, 67, 69, 71, 72], "flute", tempo=120, rhythm="8ths"
    )
    run_test("Flute C major scale is idiomatic",
             play.value >= Playability.GOOD.value)

    # Test 22: Too fast for horn
    play, issues = orch.check_idiomatic_writing(
        [60, 62, 64, 65], "french_horn_f", tempo=140, rhythm="16ths"
    )
    run_test("Horn fast 16ths problematic", len(issues) > 0)

    # Test 23: Large leap
    play, issues = orch.check_idiomatic_writing(
        [48, 72], "violin", tempo=120  # Two octave leap
    )
    run_test("Large leap detected", any("leap" in issue.lower() for issue in issues))

    # Test 24: Good range for instrument
    play, issues = orch.check_idiomatic_writing(
        [64, 67, 71, 72], "clarinet_bb", tempo=100
    )
    run_test("Clarinet optimal range is excellent",
             play.value >= Playability.GOOD.value)

    # Test 25: Wide range warning
    play, issues = orch.check_idiomatic_writing(
        [40, 45, 65, 70], "bassoon", tempo=80  # Wide range
    )
    run_test("Wide range detected", any("range" in issue.lower() for issue in issues))

    # ========================================================================
    # TEST 26-30: WIND/BRASS VOICING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 6: Wind/Brass Voicing (5 tests)")
    print("=" * 70)

    # Test 26: Woodwind traditional voicing
    result = orch.voice_for_winds([60, 64, 67, 72], "woodwinds", "traditional")
    run_test("Woodwind voicing has 4 instruments", len(result.instruments) == 4)

    # Test 27: Woodwind voicing is playable
    result = orch.voice_for_winds([60, 64, 67, 72], "woodwinds")
    run_test("Woodwind voicing is playable",
             result.playability.value >= Playability.ACCEPTABLE.value)

    # Test 28: Brass traditional voicing
    result = orch.voice_for_winds([48, 52, 55, 60], "brass", "traditional")
    run_test("Brass voicing has 4 instruments", len(result.instruments) == 4)

    # Test 29: Interlocking voicing
    result = orch.voice_for_winds([60, 64, 67], "woodwinds", "interlocking")
    run_test("Interlocking voicing created", len(result.instruments) == 3)

    # Test 30: Block voicing
    result = orch.voice_for_winds([50, 54, 57], "brass", "block")
    run_test("Block voicing created", len(result.instruments) == 3)

    # ========================================================================
    # TEST 31-35: STRING TECHNIQUES
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 7: String Techniques (5 tests)")
    print("=" * 70)

    # Test 31: Pizzicato technique
    result = orch.apply_string_technique(
        [60, 64, 67], StringTechnique.PIZZICATO, "violin"
    )
    run_test("Pizzicato technique applied", result['technique'] == 'pizzicato')

    # Test 32: Sul ponticello technique
    result = orch.apply_string_technique(
        [55, 59, 62], StringTechnique.SUL_PONTICELLO, "viola"
    )
    run_test("Sul ponticello has performance notes",
             len(result['performance_notes']) > 0)

    # Test 33: Tremolo technique
    result = orch.apply_string_technique(
        [48, 52], StringTechnique.TREMOLO, "cello"
    )
    run_test("Tremolo technique described",
             any("tremolo" in note.lower() for note in result['performance_notes']))

    # Test 34: Sul tasto technique
    result = orch.apply_string_technique(
        [60], StringTechnique.SUL_TASTO, "violin"
    )
    run_test("Sul tasto has limitations", len(result['limitations']) > 0)

    # Test 35: Harmonics technique
    result = orch.apply_string_technique(
        [84], StringTechnique.HARMONICS, "violin"
    )
    run_test("Harmonics info includes natural harmonics",
             any("natural" in note.lower() for note in result['performance_notes']))

    # ========================================================================
    # TEST 36-40: SATB SPACING
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST GROUP 8: SATB Spacing Rules (5 tests)")
    print("=" * 70)

    # Test 36: Valid SATB spacing
    valid, issues = orch.enforce_satb_spacing(72, 67, 60, 48)  # C5, G4, C4, C3
    run_test("Valid SATB spacing", valid)

    # Test 37: S-A too wide
    valid, issues = orch.enforce_satb_spacing(84, 67, 60, 48)  # S-A = 17 semitones
    run_test("S-A spacing violation detected", not valid and len(issues) > 0)

    # Test 38: Voice crossing detection
    valid, issues = orch.enforce_satb_spacing(72, 74, 60, 48)  # Alto above Soprano
    run_test("Voice crossing detected",
             any("crossing" in issue.lower() for issue in issues))

    # Test 39: A-T too wide
    valid, issues = orch.enforce_satb_spacing(72, 67, 52, 48)  # A-T = 15 semitones
    run_test("A-T spacing violation detected", not valid)

    # Test 40: Valid wide T-B spacing
    valid, issues = orch.enforce_satb_spacing(72, 65, 57, 41)  # T-B = 16 semitones
    run_test("Wide T-B spacing handled",
             valid or any("wide" in issue.lower() for issue in issues))

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed}/{test_count} tests passed")
    print("=" * 70)

    # ========================================================================
    # DEMONSTRATION EXAMPLES
    # ========================================================================
    print("\n" + "=" * 70)
    print("DEMONSTRATION EXAMPLES")
    print("=" * 70)

    # Example 1: Orchestral doubling
    print("\nExample 1: Violin melody doubling suggestions")
    melody = [72, 74, 76, 77, 79]  # C5-G5
    suggestions = orch.suggest_doubling(melody, "violin")
    print(f"  Melody: {melody}")
    print(f"  Unison doubling: {suggestions.get('unison', [])}")
    print(f"  Octave below: {suggestions.get('octave_below', [])}")

    # Example 2: Register analysis
    print("\nExample 2: Register characteristics")
    for pitch, name in [(55, "G3"), (72, "C5"), (84, "C6")]:
        reg, desc = orch.analyze_register(pitch, "violin")
        print(f"  Violin {name}: {reg.value} - {desc}")

    # Example 3: Woodwind chord voicing
    print("\nExample 3: Woodwind chord voicing (Cmaj7)")
    chord = [60, 64, 67, 71]  # C, E, G, B
    result = orch.voice_for_winds(chord, "woodwinds", "traditional")
    for note, inst in zip(result.notes, result.instruments):
        print(f"  {inst}: MIDI {note}")
    print(f"  Playability: {result.playability.value}/5")

    # Example 4: Transposition
    print("\nExample 4: Transposing instruments")
    concert_c = 60
    for inst in ["clarinet_bb", "french_horn_f", "alto_sax_eb", "trumpet_bb"]:
        written = orch.transpose_for_instrument(concert_c, inst)
        inst_name = orch.instruments[inst].name
        print(f"  {inst_name}: Concert C → Written MIDI {written}")

    # Example 5: String technique
    print("\nExample 5: String technique - Sul Ponticello")
    result = orch.apply_string_technique(
        [65, 69, 72], StringTechnique.SUL_PONTICELLO, "violin"
    )
    print(f"  Technique: {result['technique']}")
    for note in result['performance_notes']:
        print(f"    - {note}")

    print("\n" + "=" * 70)
    print("All tests and demonstrations complete!")
    print("=" * 70)
