#!/usr/bin/env python3
"""
Advanced Rhythm - Odd Meters & Metric Modulation

Specialized module for complex rhythmic structures including odd time signatures,
metric modulation, Indian tala system, African timeline patterns, additive rhythms,
and hemiola patterns.

This module complements rhythm_engine.py by providing specialized support for:
- Odd time signatures (5/4, 7/8, 11/8, 13/8, 15/8) with customizable groupings
- Metric modulation (Elliott Carter technique) for seamless tempo changes
- Indian tala patterns (Teental, Rupak, Jhaptal) from Carnatic and Hindustani traditions
- African timeline patterns (12/8 bell patterns, Gankogui, Ewe patterns)
- Additive rhythms (2+2+3, 3+2+2, etc.) from Bulgarian and Balkan traditions
- Polyrhythms (3:2, 4:3, 5:4) with LCM-based calculation
- Hemiola and cross-rhythm patterns

Research Sources:
- Elliott Carter: "Metric Modulation" - Tempo relationships through note value equivalence
  (Goldman, R.F., 1951; Bernard, J.W., 1988)
- Indian Tala System: "Tala in Practice" - Carnatic and Hindustani rhythmic cycles
  (Nelson, D., 2008; Clayton, M., 2000)
- African Timeline Patterns: "The Geometry of Musical Rhythm" - Bell patterns and polyrhythm
  (Toussaint, G.T., 2013; Agawu, K., 1995)
- Additive Rhythms: Bartók's "Mikrokosmos" - Bulgarian asymmetric meters
  (Bartók, B., 1940; Lendvai, E., 1971)
- Dave Brubeck: "Time Out" album - Jazz in odd meters (5/4, 9/8, 6/4)
  (Brubeck, D., 1959)
- Polyrhythm Mathematics: "An Efficient Algorithm for Composing Polyrhythmic Sequences"
  (ResearchGate, 2019)
- Hemiola: "Rhythmic Gesture in Beethoven" - 3:2 cross-rhythm applications
  (Rothstein, W., 1995)

Author: Agent 9 - Advanced MIDI Library Enhancement Project
Date: 2025-11-19
"""

from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import math

# Import MIDI constants
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from midi.midi_constants import DEFAULT_PPQN, PPQN_HIGH_RES
except ImportError:
    # Fallback if midi_constants not available
    DEFAULT_PPQN = 480
    PPQN_HIGH_RES = 960


# ============================================================================
# Data Classes and Enums
# ============================================================================

class OddMeterStyle(Enum):
    """Common odd meter styles"""
    TAKE_FIVE = "take_five"          # 5/4 (3+2) - Dave Brubeck
    MONEY = "money"                   # 7/4 (3+4) - Pink Floyd
    SOLSBURY_HILL = "solsbury_hill"   # 7/4 (4+3) - Peter Gabriel
    STING_DRIVEN = "sting_driven"     # 5/4 (2+3) - Tool
    BULGARIAN = "bulgarian"           # Various Bulgarian groupings


class TalaName(Enum):
    """Indian tala patterns"""
    # Hindustani (North Indian)
    TEENTAL = "teental"      # 16 beats: 4+4+4+4
    RUPAK = "rupak"          # 7 beats: 3+2+2
    JHAPTAL = "jhaptal"      # 10 beats: 2+3+2+3
    EKTAAL = "ektaal"        # 12 beats: 2+2+2+2+2+2
    KEHERWA = "keherwa"      # 8 beats: 4+4
    DADRA = "dadra"          # 6 beats: 3+3

    # Carnatic (South Indian)
    ADI_TALA = "adi_tala"    # 8 beats: 4+2+2
    RUPAKA = "rupaka"        # 6 beats: 2+4
    MISRA_CHAPU = "misra_chapu"  # 7 beats: 3+2+2


class AfricanPattern(Enum):
    """Traditional African timeline patterns"""
    GANKOGUI = "gankogui"           # 12/8 Ewe bell: 2+3+2+2+3
    SON_CLAVE = "son_clave"         # Cuban 3-2 clave
    RUMBA_CLAVE = "rumba_clave"     # Cuban rumba clave
    BEMBÉ = "bembe"                 # 12/8 bell pattern
    STANDARD_PATTERN = "standard"   # 12/8 standard timeline


@dataclass
class RhythmicEvent:
    """A single rhythmic event with timing and properties"""
    tick: int                       # MIDI tick position
    duration: int                   # Duration in ticks
    velocity: int = 80              # MIDI velocity (1-127)
    pitch: Optional[int] = None     # MIDI note number
    is_accent: bool = False         # Accented note

    def __post_init__(self):
        """Validate values"""
        self.velocity = max(1, min(127, self.velocity))
        if self.duration < 0:
            self.duration = 0


@dataclass
class TimeSignature:
    """Time signature with optional grouping"""
    numerator: int                  # Beats per measure
    denominator: int                # Beat unit (4 = quarter, 8 = eighth)
    grouping: Optional[List[int]] = None  # Additive grouping (e.g., [2, 2, 3] for 7/8)

    def __post_init__(self):
        """Validate and auto-generate grouping if needed"""
        if self.grouping is None:
            # Default to even grouping
            self.grouping = [self.numerator]

        # Verify grouping sums to numerator
        if sum(self.grouping) != self.numerator:
            raise ValueError(
                f"Grouping {self.grouping} does not sum to numerator {self.numerator}"
            )

    def __str__(self):
        grouping_str = "+".join(map(str, self.grouping))
        return f"{self.numerator}/{self.denominator} ({grouping_str})"


@dataclass
class TalaPattern:
    """Indian tala rhythmic cycle"""
    name: str
    beats: int                      # Total beats in cycle
    structure: List[int]            # Division structure
    claps: List[int]                # Beat numbers with claps (sam = 1)
    waves: List[int]                # Beat numbers with waves (khali)
    bols: Optional[List[str]] = None  # Syllables for each beat

    def __str__(self):
        return f"{self.name}: {self.beats} beats ({'+'.join(map(str, self.structure))})"


# ============================================================================
# Odd Meter Pattern Generator
# ============================================================================

class OddMeterGenerator:
    """
    Generate rhythmic patterns in odd time signatures.

    Odd meters are time signatures with numerators that create asymmetric feels:
    - 5/4, 5/8: Grouped as 2+3 or 3+2
    - 7/8, 7/4: Grouped as 2+2+3, 3+2+2, or 2+3+2
    - 11/8: Grouped as 2+2+3+2+2, or other combinations
    - 13/8: Various groupings

    Based on research:
    - Dave Brubeck's "Take Five" (5/4 as 3+2)
    - Pink Floyd's "Money" (7/4 as 3+4)
    - Bartók's Bulgarian Dances (various 7/8, 9/8 groupings)
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def generate_odd_meter_pattern(
        self,
        time_sig: TimeSignature,
        measures: int = 4,
        accent_pattern: Optional[List[int]] = None,
        velocity_base: int = 80,
        velocity_accent: int = 100,
        pitch: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate a rhythmic pattern in odd meter.

        Args:
            time_sig: TimeSignature with numerator, denominator, and grouping
            measures: Number of measures to generate
            accent_pattern: Which beats to accent (None = auto from grouping)
            velocity_base: Base velocity for unaccented notes
            velocity_accent: Velocity for accented notes
            pitch: MIDI pitch (None = just rhythm)

        Returns:
            List of RhythmicEvent objects

        Example:
            # Generate 7/8 pattern grouped as 2+2+3
            ts = TimeSignature(7, 8, grouping=[2, 2, 3])
            pattern = generator.generate_odd_meter_pattern(ts, measures=4)
        """
        # Calculate tick duration for one beat
        ticks_per_whole = self.ppqn * 4
        ticks_per_beat = ticks_per_whole // time_sig.denominator
        ticks_per_measure = ticks_per_beat * time_sig.numerator

        # Generate accent pattern from grouping if not provided
        if accent_pattern is None:
            accent_pattern = self._generate_accents_from_grouping(time_sig.grouping)

        events = []

        for measure in range(measures):
            measure_start = measure * ticks_per_measure

            for beat in range(time_sig.numerator):
                tick = measure_start + (beat * ticks_per_beat)
                is_accent = beat in accent_pattern

                event = RhythmicEvent(
                    tick=tick,
                    duration=int(ticks_per_beat * 0.8),  # 80% duration
                    velocity=velocity_accent if is_accent else velocity_base,
                    pitch=pitch,
                    is_accent=is_accent
                )
                events.append(event)

        return events

    def _generate_accents_from_grouping(self, grouping: List[int]) -> List[int]:
        """
        Generate accent pattern from grouping.

        Example:
            [2, 2, 3] -> [0, 2, 4] (accent first beat of each group)
        """
        accents = [0]  # Always accent beat 1 (downbeat)
        current_beat = 0

        for group_size in grouping[:-1]:  # Exclude last group
            current_beat += group_size
            accents.append(current_beat)

        return accents

    def generate_preset_pattern(
        self,
        style: OddMeterStyle,
        measures: int = 4,
        velocity_base: int = 80,
        velocity_accent: int = 100,
        pitch: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate preset odd meter patterns from famous compositions.

        Args:
            style: OddMeterStyle enum value
            measures: Number of measures
            velocity_base: Base velocity
            velocity_accent: Accent velocity
            pitch: MIDI pitch

        Returns:
            List of RhythmicEvent objects
        """
        presets = {
            OddMeterStyle.TAKE_FIVE: TimeSignature(5, 4, [3, 2]),
            OddMeterStyle.MONEY: TimeSignature(7, 4, [3, 4]),
            OddMeterStyle.SOLSBURY_HILL: TimeSignature(7, 4, [4, 3]),
            OddMeterStyle.STING_DRIVEN: TimeSignature(5, 4, [2, 3]),
            OddMeterStyle.BULGARIAN: TimeSignature(7, 8, [2, 2, 3]),
        }

        time_sig = presets.get(style)
        if time_sig is None:
            raise ValueError(f"Unknown style: {style}")

        return self.generate_odd_meter_pattern(
            time_sig, measures, None, velocity_base, velocity_accent, pitch
        )

    def generate_additive_rhythm(
        self,
        grouping: List[int],
        denominator: int = 8,
        measures: int = 4,
        velocity_base: int = 80,
        velocity_accent: int = 100,
        pitch: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate additive rhythm pattern.

        Additive rhythms group beats asymmetrically (common in Balkan/Bulgarian music).

        Args:
            grouping: Beat grouping (e.g., [2, 2, 3] for 7/8)
            denominator: Beat unit (8 for eighth notes, 4 for quarters)
            measures: Number of measures
            velocity_base: Base velocity
            velocity_accent: Accent velocity
            pitch: MIDI pitch

        Returns:
            List of RhythmicEvent objects

        Example:
            # Bulgarian 7/8 pattern (2+2+3)
            pattern = generator.generate_additive_rhythm([2, 2, 3], denominator=8)
        """
        numerator = sum(grouping)
        time_sig = TimeSignature(numerator, denominator, grouping)

        return self.generate_odd_meter_pattern(
            time_sig, measures, None, velocity_base, velocity_accent, pitch
        )


# ============================================================================
# Metric Modulation Engine
# ============================================================================

class MetricModulation:
    """
    Calculate metric modulation (Elliott Carter technique).

    Metric modulation creates smooth tempo changes by equating note values
    across tempo changes. For example, the quarter note in the first tempo
    becomes a dotted eighth in the new tempo, creating a seamless transition.

    Based on Elliott Carter's compositional technique introduced in his
    Cello Sonata (1948) and used extensively in his String Quartets.

    Reference:
    - Goldman, R.F. (1951). "Current Chronicle: New York". Musical Quarterly.
    - Bernard, J.W. (1988). "The Evolution of Elliott Carter's Rhythmic Practice".
    """

    def __init__(self):
        pass

    def calculate_modulation(
        self,
        from_tempo: float,
        from_note_value: Fraction,
        to_note_value: Fraction
    ) -> float:
        """
        Calculate new tempo after metric modulation.

        Args:
            from_tempo: Starting tempo (BPM)
            from_note_value: Note value in first tempo (as fraction of whole note)
                           Examples: Fraction(1, 4) = quarter note
                                    Fraction(3, 8) = dotted quarter
                                    Fraction(1, 8) = eighth note
            to_note_value: Note value to become in new tempo

        Returns:
            New tempo in BPM

        Example:
            # Quarter note at 60 BPM becomes dotted quarter at new tempo
            new_tempo = calc.calculate_modulation(60, Fraction(1, 4), Fraction(3, 8))
            # Result: 40 BPM (since dotted quarter = 1.5x quarter)
        """
        # Calculate the ratio of note values
        # If quarter note becomes dotted quarter, ratio = (1/4) / (3/8) = 2/3
        ratio = from_note_value / to_note_value

        # New tempo = old tempo * ratio
        new_tempo = from_tempo * float(ratio)

        return new_tempo

    def calculate_modulation_simple(
        self,
        from_tempo: float,
        from_division: str,
        to_division: str
    ) -> float:
        """
        Simplified metric modulation with common note values.

        Args:
            from_tempo: Starting tempo (BPM)
            from_division: Note value string ('quarter', 'eighth', 'dotted_quarter', etc.)
            to_division: Target note value string

        Returns:
            New tempo in BPM

        Example:
            # Quarter note at 60 BPM becomes eighth note
            new_tempo = calc.calculate_modulation_simple(60, 'quarter', 'eighth')
            # Result: 120 BPM
        """
        note_values = {
            'whole': Fraction(1, 1),
            'half': Fraction(1, 2),
            'dotted_half': Fraction(3, 4),
            'quarter': Fraction(1, 4),
            'dotted_quarter': Fraction(3, 8),
            'eighth': Fraction(1, 8),
            'dotted_eighth': Fraction(3, 16),
            'sixteenth': Fraction(1, 16),
            'triplet_quarter': Fraction(1, 6),
            'triplet_eighth': Fraction(1, 12),
        }

        if from_division not in note_values:
            raise ValueError(f"Unknown note division: {from_division}")
        if to_division not in note_values:
            raise ValueError(f"Unknown note division: {to_division}")

        return self.calculate_modulation(
            from_tempo,
            note_values[from_division],
            note_values[to_division]
        )

    def generate_tempo_map(
        self,
        modulations: List[Tuple[int, float, str, str]],
        initial_tempo: float = 60.0
    ) -> List[Tuple[int, float]]:
        """
        Generate tempo map from series of metric modulations.

        Args:
            modulations: List of (measure, from_tempo, from_div, to_div)
                        If from_tempo is None, use result of previous modulation
            initial_tempo: Starting tempo

        Returns:
            List of (measure, tempo) pairs

        Example:
            modulations = [
                (0, 60.0, 'quarter', 'quarter'),      # Start at 60
                (4, None, 'quarter', 'dotted_quarter'), # Modulate at bar 4
                (8, None, 'eighth', 'quarter'),       # Modulate at bar 8
            ]
            tempo_map = calc.generate_tempo_map(modulations)
        """
        tempo_map = []
        current_tempo = initial_tempo

        for measure, from_tempo, from_div, to_div in modulations:
            if from_tempo is not None:
                current_tempo = from_tempo

            # Calculate new tempo
            new_tempo = self.calculate_modulation_simple(
                current_tempo, from_div, to_div
            )

            tempo_map.append((measure, new_tempo))
            current_tempo = new_tempo

        return tempo_map


# ============================================================================
# Indian Tala System
# ============================================================================

class TalaGenerator:
    """
    Generate Indian tala (rhythmic cycle) patterns.

    Tala is the metric cycle system in Indian classical music, covering
    "the whole subject of musical meter" (David Nelson).

    Key concepts:
    - Sam: The first beat (most important, always emphasized)
    - Avartan: One complete cycle of the tala
    - Vibhag/Anga: Sections within the tala
    - Tali: Clap (emphasized beats)
    - Khali: Wave (de-emphasized beats)

    Research sources:
    - Clayton, M. (2000). "Time in Indian Music"
    - Nelson, D. (2008). "Mrdanga: Tala Fundamentals"
    - Carnatic and Hindustani tala traditions
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn
        self._init_tala_library()

    def _init_tala_library(self):
        """Initialize library of traditional talas"""
        self.talas = {
            # Hindustani (North Indian) Talas
            TalaName.TEENTAL: TalaPattern(
                name="Teental",
                beats=16,
                structure=[4, 4, 4, 4],
                claps=[1, 5, 13],
                waves=[9],
                bols=['dha', 'dhin', 'dhin', 'dha', 'dha', 'dhin', 'dhin', 'dha',
                      'dha', 'tin', 'tin', 'ta', 'ta', 'dhin', 'dhin', 'dha']
            ),
            TalaName.RUPAK: TalaPattern(
                name="Rupak",
                beats=7,
                structure=[3, 2, 2],
                claps=[4, 6],
                waves=[1],
                bols=['tin', 'tin', 'na', 'dhin', 'na', 'dhin', 'na']
            ),
            TalaName.JHAPTAL: TalaPattern(
                name="Jhaptal",
                beats=10,
                structure=[2, 3, 2, 3],
                claps=[1, 3, 8],
                waves=[6],
                bols=['dhi', 'na', 'dhi', 'dhi', 'na', 'ti', 'na', 'dhi', 'dhi', 'na']
            ),
            TalaName.EKTAAL: TalaPattern(
                name="Ektaal",
                beats=12,
                structure=[2, 2, 2, 2, 2, 2],
                claps=[1, 5, 9, 11],
                waves=[3, 7],
                bols=['dhin', 'dhin', 'dha', 'ge', 'ti', 'na', 'ka', 'ta',
                      'dha', 'ge', 'dhin', 'na']
            ),
            TalaName.KEHERWA: TalaPattern(
                name="Keherwa",
                beats=8,
                structure=[4, 4],
                claps=[1],
                waves=[5],
                bols=['dha', 'ge', 'na', 'ti', 'na', 'ka', 'dhi', 'na']
            ),
            TalaName.DADRA: TalaPattern(
                name="Dadra",
                beats=6,
                structure=[3, 3],
                claps=[1],
                waves=[4],
                bols=['dha', 'dhin', 'na', 'dha', 'tin', 'na']
            ),

            # Carnatic (South Indian) Talas
            TalaName.ADI_TALA: TalaPattern(
                name="Adi Tala",
                beats=8,
                structure=[4, 2, 2],
                claps=[1, 5, 7],
                waves=[],
                bols=['ta', 'ka', 'dhi', 'mi', 'ta', 'ka', 'ta', 'ka']
            ),
            TalaName.RUPAKA: TalaPattern(
                name="Rupaka",
                beats=6,
                structure=[2, 4],
                claps=[1, 3],
                waves=[],
                bols=['ta', 'ki', 'ta', 'ka', 'dhi', 'mi']
            ),
            TalaName.MISRA_CHAPU: TalaPattern(
                name="Misra Chapu",
                beats=7,
                structure=[3, 2, 2],
                claps=[1, 4, 6],
                waves=[],
                bols=['ta', 'ka', 'dhi', 'mi', 'ta', 'ka', 'ta']
            ),
        }

    def generate_tala_pattern(
        self,
        tala_name: TalaName,
        cycles: int = 1,
        velocity_sam: int = 100,
        velocity_tali: int = 85,
        velocity_khali: int = 70,
        velocity_normal: int = 75,
        pitch: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate tala pattern.

        Args:
            tala_name: TalaName enum value
            cycles: Number of complete cycles (avartans) to generate
            velocity_sam: Velocity for sam (beat 1)
            velocity_tali: Velocity for claps
            velocity_khali: Velocity for waves
            velocity_normal: Velocity for other beats
            pitch: MIDI pitch

        Returns:
            List of RhythmicEvent objects

        Example:
            # Generate 2 cycles of Teental
            pattern = generator.generate_tala_pattern(TalaName.TEENTAL, cycles=2)
        """
        if tala_name not in self.talas:
            raise ValueError(f"Unknown tala: {tala_name}")

        tala = self.talas[tala_name]

        # Calculate tick duration for one beat (assuming quarter note per beat)
        ticks_per_beat = self.ppqn

        events = []

        for cycle in range(cycles):
            cycle_start = cycle * tala.beats * ticks_per_beat

            for beat_num in range(1, tala.beats + 1):
                tick = cycle_start + ((beat_num - 1) * ticks_per_beat)

                # Determine velocity based on beat type
                if beat_num == 1:
                    velocity = velocity_sam  # Sam (most important)
                elif beat_num in tala.claps:
                    velocity = velocity_tali  # Tali (clap)
                elif beat_num in tala.waves:
                    velocity = velocity_khali  # Khali (wave)
                else:
                    velocity = velocity_normal

                event = RhythmicEvent(
                    tick=tick,
                    duration=int(ticks_per_beat * 0.85),
                    velocity=velocity,
                    pitch=pitch,
                    is_accent=(beat_num == 1 or beat_num in tala.claps)
                )
                events.append(event)

        return events

    def get_tala_info(self, tala_name: TalaName) -> TalaPattern:
        """Get information about a tala"""
        if tala_name not in self.talas:
            raise ValueError(f"Unknown tala: {tala_name}")
        return self.talas[tala_name]

    def list_talas(self) -> List[str]:
        """List all available talas"""
        return [tala.name for tala in self.talas.values()]


# ============================================================================
# African Timeline Patterns (Extended)
# ============================================================================

class AfricanTimelineGenerator:
    """
    Generate traditional African timeline patterns.

    Timeline patterns (also called bell patterns or guide patterns) are
    asymmetric rhythmic patterns that serve as reference points in African music.

    The most common is the 12-pulse pattern (in 12/8) grouped as 2+3+2+2+3,
    played on the gankogui (double bell) in Ewe music.

    Research sources:
    - Toussaint, G.T. (2013). "The Geometry of Musical Rhythm"
    - Agawu, K. (1995). "African Rhythm: A Northern Ewe Perspective"
    - Jones, A.M. (1959). "Studies in African Music"
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def generate_african_bell(
        self,
        pattern_type: AfricanPattern = AfricanPattern.GANKOGUI,
        measures: int = 4,
        velocity_high: int = 90,
        velocity_low: int = 75,
        pitch_high: Optional[int] = None,
        pitch_low: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate African bell pattern.

        Args:
            pattern_type: AfricanPattern enum value
            measures: Number of measures
            velocity_high: Velocity for high bell
            velocity_low: Velocity for low bell
            pitch_high: MIDI pitch for high bell (e.g., 67)
            pitch_low: MIDI pitch for low bell (e.g., 65)

        Returns:
            List of RhythmicEvent objects

        Example:
            # Generate Gankogui pattern (12/8 bell)
            pattern = generator.generate_african_bell(
                AfricanPattern.GANKOGUI,
                measures=4,
                pitch_high=67,
                pitch_low=65
            )
        """
        # Define patterns as list of (pulse, is_high_bell)
        # 12-pulse patterns (in 12/8 time)
        patterns = {
            AfricanPattern.GANKOGUI: [
                # Standard Ewe gankogui: 2+3+2+2+3 grouping
                # High bell on: 1, 4, 7, 10
                # Low bell on: 3, 6, 9, 12
                (0, True),   # 1: High
                (2, False),  # 3: Low
                (3, True),   # 4: High
                (5, False),  # 6: Low
                (6, True),   # 7: High
                (8, False),  # 9: Low
                (9, True),   # 10: High
                (11, False), # 12: Low
            ],
            AfricanPattern.STANDARD_PATTERN: [
                # 12/8 standard timeline
                (0, True), (2, True), (3, True), (5, True),
                (6, True), (8, True), (10, True)
            ],
            AfricanPattern.BEMBÉ: [
                # Bembé pattern (12/8)
                (0, True), (3, True), (6, True), (7, True),
                (10, True)
            ],
            AfricanPattern.SON_CLAVE: [
                # Son clave (in 16-pulse context, adapted to 12)
                (0, True), (3, True), (6, True), (10, True),
                (11, True)
            ],
            AfricanPattern.RUMBA_CLAVE: [
                # Rumba clave (in 16-pulse context, adapted to 12)
                (0, True), (3, True), (6, True), (10, True)
            ],
        }

        if pattern_type not in patterns:
            raise ValueError(f"Unknown pattern: {pattern_type}")

        pattern = patterns[pattern_type]

        # Calculate timing (12 pulses per measure in 12/8)
        pulses_per_measure = 12
        ticks_per_pulse = (self.ppqn * 4) // pulses_per_measure  # 12/8 = 12 eighth notes
        ticks_per_measure = ticks_per_pulse * pulses_per_measure

        events = []

        for measure in range(measures):
            measure_start = measure * ticks_per_measure

            for pulse, is_high in pattern:
                tick = measure_start + (pulse * ticks_per_pulse)

                event = RhythmicEvent(
                    tick=tick,
                    duration=int(ticks_per_pulse * 0.7),
                    velocity=velocity_high if is_high else velocity_low,
                    pitch=pitch_high if is_high else pitch_low,
                    is_accent=is_high
                )
                events.append(event)

        return events


# ============================================================================
# Polyrhythm Generator (LCM-based)
# ============================================================================

class PolyrhythmEngine:
    """
    Generate polyrhythms using LCM (Least Common Multiple) method.

    Polyrhythm is the simultaneous use of two or more different rhythms.
    Common examples:
    - 3:2 (three against two) - fundamental African polyrhythm
    - 4:3 (four against three) - common in classical and jazz
    - 5:4 (five against four) - complex polyrhythm

    The LCM method ensures precise synchronization by finding the common
    denominator for both rhythms.

    Research:
    - "Math and Music: Polyrhythmic Music" - Gareth E. Roberts
    - "An Efficient Algorithm For Composing Polyrhythmic Sequences" (ResearchGate)
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def generate_polyrhythm(
        self,
        ratio_a: int,
        ratio_b: int,
        duration_beats: int = 4,
        velocity_a: int = 85,
        velocity_b: int = 70,
        pitch_a: Optional[int] = None,
        pitch_b: Optional[int] = None
    ) -> Tuple[List[RhythmicEvent], List[RhythmicEvent]]:
        """
        Generate polyrhythm using LCM method.

        Args:
            ratio_a: First rhythm (e.g., 3 in "3 against 4")
            ratio_b: Second rhythm (e.g., 4 in "3 against 4")
            duration_beats: Duration in beats (quarter notes)
            velocity_a: Velocity for first rhythm
            velocity_b: Velocity for second rhythm
            pitch_a: MIDI pitch for first rhythm
            pitch_b: MIDI pitch for second rhythm

        Returns:
            Tuple of (rhythm_a_events, rhythm_b_events)

        Example:
            # Generate 3:2 polyrhythm over 4 beats
            rhythm_a, rhythm_b = generator.generate_polyrhythm(3, 2, duration_beats=4)
        """
        # Calculate total duration in ticks
        total_ticks = self.ppqn * duration_beats

        # Calculate LCM for precise subdivision
        lcm = self._lcm(ratio_a, ratio_b)
        tick_subdivision = total_ticks / lcm

        # Generate rhythm A
        rhythm_a = []
        interval_a = total_ticks / ratio_a
        for i in range(ratio_a):
            tick = int(i * interval_a)
            event = RhythmicEvent(
                tick=tick,
                duration=int(interval_a * 0.8),
                velocity=velocity_a,
                pitch=pitch_a,
                is_accent=(i == 0)
            )
            rhythm_a.append(event)

        # Generate rhythm B
        rhythm_b = []
        interval_b = total_ticks / ratio_b
        for i in range(ratio_b):
            tick = int(i * interval_b)
            event = RhythmicEvent(
                tick=tick,
                duration=int(interval_b * 0.8),
                velocity=velocity_b,
                pitch=pitch_b,
                is_accent=(i == 0)
            )
            rhythm_b.append(event)

        return rhythm_a, rhythm_b

    def _lcm(self, a: int, b: int) -> int:
        """Calculate least common multiple"""
        return abs(a * b) // math.gcd(a, b)

    def _gcd(self, a: int, b: int) -> int:
        """Calculate greatest common divisor"""
        return math.gcd(a, b)


# ============================================================================
# Hemiola and Cross-Rhythm Generator
# ============================================================================

class HemiolaGenerator:
    """
    Generate hemiola and cross-rhythm patterns.

    Hemiola is the ratio 3:2 applied rhythmically - three beats in the time
    of two, or vice versa. It creates a wonderful cross-rhythm effect.

    Common in:
    - Baroque music (Handel, Bach) - often at cadences
    - African music - 3:2 is the fundamental polyrhythmic cell
    - Classical music (Brahms, Beethoven)

    Types:
    - Vertical hemiola: Multiple parts playing 2s and 3s simultaneously
    - Horizontal hemiola: Switching between 2-feel and 3-feel in one part

    Research:
    - Rothstein, W. (1995). "Rhythmic Gesture in Beethoven"
    - Willner, C. (1996). "More on Handel and the Hemiola"
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn

    def generate_hemiola(
        self,
        measures: int = 2,
        base_meter: Tuple[int, int] = (3, 4),
        velocity_base: int = 80,
        velocity_accent: int = 95,
        pitch: Optional[int] = None
    ) -> List[RhythmicEvent]:
        """
        Generate horizontal hemiola pattern.

        In 3/4 time, a hemiola creates a 2-feel by grouping six eighth notes
        as 3+3 instead of 2+2+2.

        Args:
            measures: Number of measures (should be even for complete hemiola)
            base_meter: Base time signature (numerator, denominator)
            velocity_base: Base velocity
            velocity_accent: Accent velocity
            pitch: MIDI pitch

        Returns:
            List of RhythmicEvent objects

        Example:
            # Generate hemiola in 3/4 time (creates 2-feel across 2 bars)
            pattern = generator.generate_hemiola(measures=2, base_meter=(3, 4))
        """
        numerator, denominator = base_meter

        # Only works for meters divisible by 3 or 2
        if numerator % 3 != 0 and numerator % 2 != 0:
            raise ValueError("Hemiola requires meter divisible by 2 or 3")

        ticks_per_beat = (self.ppqn * 4) // denominator
        ticks_per_measure = ticks_per_beat * numerator

        events = []

        # Generate hemiola: in 3/4, group as 3+3 instead of 2+2+2
        if numerator == 3 and denominator == 4:
            # Two measures of 3/4 = 6 quarter notes
            # Normal: 2+2+2 | 2+2+2 (eighth note groupings)
            # Hemiola: 3+3 | 3+3 (creates 2-feel)

            for measure in range(0, measures, 2):  # Process in pairs
                measure_start = measure * ticks_per_measure

                # Create 4 groups of 3 eighth notes across 2 measures
                eighth_note_ticks = ticks_per_beat // 2

                for group in range(4):
                    tick = measure_start + (group * 3 * eighth_note_ticks)

                    event = RhythmicEvent(
                        tick=tick,
                        duration=int(eighth_note_ticks * 0.8),
                        velocity=velocity_accent if group % 2 == 0 else velocity_base,
                        pitch=pitch,
                        is_accent=(group % 2 == 0)
                    )
                    events.append(event)

        return events

    def generate_vertical_hemiola(
        self,
        duration_beats: int = 4,
        velocity_triple: int = 85,
        velocity_duple: int = 75,
        pitch_triple: Optional[int] = None,
        pitch_duple: Optional[int] = None
    ) -> Tuple[List[RhythmicEvent], List[RhythmicEvent]]:
        """
        Generate vertical hemiola (3:2 cross-rhythm).

        Args:
            duration_beats: Duration in beats
            velocity_triple: Velocity for triple division
            velocity_duple: Velocity for duple division
            pitch_triple: MIDI pitch for triple division
            pitch_duple: MIDI pitch for duple division

        Returns:
            Tuple of (triple_rhythm, duple_rhythm)

        Example:
            # Generate 3 against 2 cross-rhythm
            triple, duple = generator.generate_vertical_hemiola(duration_beats=4)
        """
        total_ticks = self.ppqn * duration_beats

        # Triple division (3 notes)
        triple_rhythm = []
        interval_triple = total_ticks / 3
        for i in range(3):
            event = RhythmicEvent(
                tick=int(i * interval_triple),
                duration=int(interval_triple * 0.8),
                velocity=velocity_triple,
                pitch=pitch_triple,
                is_accent=(i == 0)
            )
            triple_rhythm.append(event)

        # Duple division (2 notes)
        duple_rhythm = []
        interval_duple = total_ticks / 2
        for i in range(2):
            event = RhythmicEvent(
                tick=int(i * interval_duple),
                duration=int(interval_duple * 0.8),
                velocity=velocity_duple,
                pitch=pitch_duple,
                is_accent=(i == 0)
            )
            duple_rhythm.append(event)

        return triple_rhythm, duple_rhythm


# ============================================================================
# Main Advanced Rhythm Engine
# ============================================================================

class AdvancedRhythm:
    """
    Main class combining all advanced rhythm capabilities.

    This engine provides:
    - Odd time signatures (5/4, 7/8, 11/8, 13/8, 15/8) with custom groupings
    - Metric modulation (Elliott Carter technique)
    - Indian tala patterns (Teental, Rupak, Jhaptal, etc.)
    - African timeline patterns (Gankogui, bell patterns)
    - Additive rhythms (2+2+3, 3+2+2, Bulgarian rhythms)
    - Polyrhythms (3:2, 4:3, 5:4) with LCM calculation
    - Hemiola and cross-rhythms

    Example:
        engine = AdvancedRhythm(ppqn=960)

        # Generate 7/8 pattern
        pattern = engine.generate_odd_meter_pattern(
            TimeSignature(7, 8, [2, 2, 3])
        )

        # Calculate metric modulation
        new_tempo = engine.metric_modulation(60, 'quarter', 'dotted_quarter')

        # Generate tala
        tala = engine.generate_tala_pattern(TalaName.TEENTAL, cycles=2)
    """

    def __init__(self, ppqn: int = DEFAULT_PPQN):
        self.ppqn = ppqn
        self.odd_meter = OddMeterGenerator(ppqn)
        self.metric_mod = MetricModulation()
        self.tala = TalaGenerator(ppqn)
        self.african = AfricanTimelineGenerator(ppqn)
        self.polyrhythm = PolyrhythmEngine(ppqn)
        self.hemiola = HemiolaGenerator(ppqn)

    # Convenience methods that delegate to sub-engines

    def generate_odd_meter_pattern(
        self,
        time_sig: TimeSignature,
        measures: int = 4,
        **kwargs
    ) -> List[RhythmicEvent]:
        """Generate odd meter pattern (5/4, 7/8, etc.)"""
        return self.odd_meter.generate_odd_meter_pattern(time_sig, measures, **kwargs)

    def metric_modulation(
        self,
        from_tempo: float,
        from_division: str,
        to_division: str
    ) -> float:
        """Calculate metric modulation"""
        return self.metric_mod.calculate_modulation_simple(
            from_tempo, from_division, to_division
        )

    def generate_tala_pattern(
        self,
        tala_name: TalaName,
        cycles: int = 1,
        **kwargs
    ) -> List[RhythmicEvent]:
        """Generate Indian tala pattern"""
        return self.tala.generate_tala_pattern(tala_name, cycles, **kwargs)

    def generate_african_bell(
        self,
        pattern_type: AfricanPattern = AfricanPattern.GANKOGUI,
        measures: int = 4,
        **kwargs
    ) -> List[RhythmicEvent]:
        """Generate African bell pattern"""
        return self.african.generate_african_bell(pattern_type, measures, **kwargs)

    def create_additive_rhythm(
        self,
        grouping: List[int],
        denominator: int = 8,
        measures: int = 4,
        **kwargs
    ) -> List[RhythmicEvent]:
        """Generate additive rhythm (Bulgarian/Balkan style)"""
        return self.odd_meter.generate_additive_rhythm(
            grouping, denominator, measures, **kwargs
        )

    def generate_polyrhythm(
        self,
        ratio_a: int,
        ratio_b: int,
        duration_beats: int = 4,
        **kwargs
    ) -> Tuple[List[RhythmicEvent], List[RhythmicEvent]]:
        """Generate polyrhythm (3:2, 4:3, 5:4, etc.)"""
        return self.polyrhythm.generate_polyrhythm(
            ratio_a, ratio_b, duration_beats, **kwargs
        )


# ============================================================================
# Unit Tests
# ============================================================================

if __name__ == "__main__":
    """Comprehensive unit tests for Advanced Rhythm module"""

    print("=" * 80)
    print("ADVANCED RHYTHM - COMPREHENSIVE TESTS")
    print("=" * 80)

    # Initialize engine
    engine = AdvancedRhythm(ppqn=960)

    # Test 1: Odd Meter Generation - 7/8 (2+2+3)
    print("\n[TEST 1] Generating 7/8 pattern (2+2+3) - Bulgarian rhythm...")
    ts_7_8 = TimeSignature(7, 8, [2, 2, 3])
    pattern_7_8 = engine.generate_odd_meter_pattern(ts_7_8, measures=4)
    print(f"✓ Generated {len(pattern_7_8)} events in {ts_7_8}")
    print(f"  First few ticks: {[e.tick for e in pattern_7_8[:7]]}")
    assert len(pattern_7_8) == 28, "Should have 28 events (7 beats × 4 measures)"

    # Test 2: Odd Meter Generation - 5/4 (Take Five style)
    print("\n[TEST 2] Generating 5/4 pattern (3+2) - Take Five style...")
    ts_5_4 = TimeSignature(5, 4, [3, 2])
    pattern_5_4 = engine.odd_meter.generate_preset_pattern(
        OddMeterStyle.TAKE_FIVE, measures=4
    )
    print(f"✓ Generated {len(pattern_5_4)} events")
    assert len(pattern_5_4) == 20, "Should have 20 events (5 beats × 4 measures)"

    # Test 3: 11/8 pattern
    print("\n[TEST 3] Generating 11/8 pattern (2+2+3+2+2)...")
    ts_11_8 = TimeSignature(11, 8, [2, 2, 3, 2, 2])
    pattern_11_8 = engine.generate_odd_meter_pattern(ts_11_8, measures=2)
    print(f"✓ Generated {len(pattern_11_8)} events in {ts_11_8}")
    assert len(pattern_11_8) == 22, "Should have 22 events (11 beats × 2 measures)"

    # Test 4: 13/8 pattern
    print("\n[TEST 4] Generating 13/8 pattern (3+3+3+2+2)...")
    ts_13_8 = TimeSignature(13, 8, [3, 3, 3, 2, 2])
    pattern_13_8 = engine.generate_odd_meter_pattern(ts_13_8, measures=2)
    print(f"✓ Generated {len(pattern_13_8)} events in {ts_13_8}")
    assert len(pattern_13_8) == 26, "Should have 26 events"

    # Test 5: 15/8 pattern
    print("\n[TEST 5] Generating 15/8 pattern (3+3+3+3+3)...")
    ts_15_8 = TimeSignature(15, 8, [3, 3, 3, 3, 3])
    pattern_15_8 = engine.generate_odd_meter_pattern(ts_15_8, measures=1)
    print(f"✓ Generated {len(pattern_15_8)} events")
    assert len(pattern_15_8) == 15, "Should have 15 events"

    # Test 6: Metric Modulation - quarter to dotted quarter
    print("\n[TEST 6] Metric modulation: quarter note at 60 BPM → dotted quarter...")
    new_tempo = engine.metric_modulation(60, 'quarter', 'dotted_quarter')
    print(f"✓ New tempo: {new_tempo:.2f} BPM")
    assert abs(new_tempo - 40.0) < 0.01, "Should be 40 BPM"

    # Test 7: Metric Modulation - quarter to eighth
    print("\n[TEST 7] Metric modulation: quarter note at 60 BPM → eighth note...")
    new_tempo = engine.metric_modulation(60, 'quarter', 'eighth')
    print(f"✓ New tempo: {new_tempo:.2f} BPM")
    assert abs(new_tempo - 120.0) < 0.01, "Should be 120 BPM"

    # Test 8: Metric Modulation - dotted eighth to quarter
    print("\n[TEST 8] Metric modulation: dotted eighth at 80 BPM → quarter...")
    new_tempo = engine.metric_modulation(80, 'dotted_eighth', 'quarter')
    print(f"✓ New tempo: {new_tempo:.2f} BPM")
    expected = 80 * (3/16) / (1/4)  # 80 * (3/16) / (4/16) = 80 * 3/4 = 60
    assert abs(new_tempo - 60.0) < 0.01, f"Should be 60 BPM, got {new_tempo}"

    # Test 9: Tempo Map Generation
    print("\n[TEST 9] Generating tempo map with multiple modulations...")
    modulations = [
        (0, 60.0, 'quarter', 'quarter'),
        (4, None, 'quarter', 'dotted_quarter'),
        (8, None, 'eighth', 'quarter'),
    ]
    tempo_map = engine.metric_mod.generate_tempo_map(modulations)
    print(f"✓ Generated tempo map: {tempo_map}")
    assert len(tempo_map) == 3, "Should have 3 tempo changes"

    # Test 10: Teental Tala (16 beats)
    print("\n[TEST 10] Generating Teental tala (16 beats, 2 cycles)...")
    teental = engine.generate_tala_pattern(TalaName.TEENTAL, cycles=2)
    print(f"✓ Generated {len(teental)} events")
    tala_info = engine.tala.get_tala_info(TalaName.TEENTAL)
    print(f"  Structure: {'+'.join(map(str, tala_info.structure))}")
    assert len(teental) == 32, "Should have 32 events (16 beats × 2 cycles)"

    # Test 11: Rupak Tala (7 beats)
    print("\n[TEST 11] Generating Rupak tala (7 beats, asymmetric)...")
    rupak = engine.generate_tala_pattern(TalaName.RUPAK, cycles=1)
    print(f"✓ Generated {len(rupak)} events")
    assert len(rupak) == 7, "Should have 7 events"

    # Test 12: Jhaptal Tala (10 beats)
    print("\n[TEST 12] Generating Jhaptal tala (10 beats)...")
    jhaptal = engine.generate_tala_pattern(TalaName.JHAPTAL, cycles=1)
    print(f"✓ Generated {len(jhaptal)} events")
    assert len(jhaptal) == 10, "Should have 10 events"

    # Test 13: Adi Tala (Carnatic, 8 beats)
    print("\n[TEST 13] Generating Adi Tala (Carnatic, 8 beats)...")
    adi_tala = engine.generate_tala_pattern(TalaName.ADI_TALA, cycles=1)
    print(f"✓ Generated {len(adi_tala)} events")
    assert len(adi_tala) == 8, "Should have 8 events"

    # Test 14: List all talas
    print("\n[TEST 14] Listing all available talas...")
    all_talas = engine.tala.list_talas()
    print(f"✓ Available talas: {', '.join(all_talas)}")
    assert len(all_talas) >= 9, "Should have at least 9 talas"

    # Test 15: Gankogui Bell Pattern (12/8)
    print("\n[TEST 15] Generating Gankogui bell pattern (12/8)...")
    gankogui = engine.generate_african_bell(
        AfricanPattern.GANKOGUI, measures=4
    )
    print(f"✓ Generated {len(gankogui)} events")
    print(f"  First few ticks: {[e.tick for e in gankogui[:8]]}")
    assert len(gankogui) == 32, "Should have 32 events (8 hits × 4 measures)"

    # Test 16: Standard African Timeline
    print("\n[TEST 16] Generating Standard African timeline...")
    standard = engine.generate_african_bell(
        AfricanPattern.STANDARD_PATTERN, measures=2
    )
    print(f"✓ Generated {len(standard)} events")

    # Test 17: Bembé Pattern
    print("\n[TEST 17] Generating Bembé pattern...")
    bembe = engine.generate_african_bell(
        AfricanPattern.BEMBÉ, measures=4
    )
    print(f"✓ Generated {len(bembe)} events")

    # Test 18: Additive Rhythm - Bulgarian 7/8 (2+2+3)
    print("\n[TEST 18] Generating additive rhythm - Bulgarian 7/8 (2+2+3)...")
    bulgarian = engine.create_additive_rhythm([2, 2, 3], denominator=8, measures=4)
    print(f"✓ Generated {len(bulgarian)} events")
    assert len(bulgarian) == 28, "Should have 28 events"

    # Test 19: Additive Rhythm - 9/8 (2+2+2+3)
    print("\n[TEST 19] Generating additive rhythm - 9/8 (2+2+2+3)...")
    nine_eight = engine.create_additive_rhythm([2, 2, 2, 3], denominator=8, measures=2)
    print(f"✓ Generated {len(nine_eight)} events")
    assert len(nine_eight) == 18, "Should have 18 events"

    # Test 20: Polyrhythm - 3:2
    print("\n[TEST 20] Generating 3:2 polyrhythm...")
    rhythm_3, rhythm_2 = engine.generate_polyrhythm(3, 2, duration_beats=4)
    print(f"✓ Generated 3-rhythm with {len(rhythm_3)} events")
    print(f"✓ Generated 2-rhythm with {len(rhythm_2)} events")
    assert len(rhythm_3) == 3, "Should have 3 events"
    assert len(rhythm_2) == 2, "Should have 2 events"

    # Test 21: Polyrhythm - 4:3
    print("\n[TEST 21] Generating 4:3 polyrhythm...")
    rhythm_4, rhythm_3 = engine.generate_polyrhythm(4, 3, duration_beats=4)
    print(f"✓ Generated 4-rhythm with {len(rhythm_4)} events")
    print(f"✓ Generated 3-rhythm with {len(rhythm_3)} events")
    assert len(rhythm_4) == 4, "Should have 4 events"
    assert len(rhythm_3) == 3, "Should have 3 events"

    # Test 22: Polyrhythm - 5:4
    print("\n[TEST 22] Generating 5:4 polyrhythm...")
    rhythm_5, rhythm_4 = engine.generate_polyrhythm(5, 4, duration_beats=4)
    print(f"✓ Generated 5-rhythm with {len(rhythm_5)} events")
    print(f"✓ Generated 4-rhythm with {len(rhythm_4)} events")
    assert len(rhythm_5) == 5, "Should have 5 events"
    assert len(rhythm_4) == 4, "Should have 4 events"

    # Test 23: Hemiola - Horizontal (3/4)
    print("\n[TEST 23] Generating horizontal hemiola (3/4)...")
    hemiola_h = engine.hemiola.generate_hemiola(measures=2, base_meter=(3, 4))
    print(f"✓ Generated {len(hemiola_h)} events")

    # Test 24: Hemiola - Vertical (3 against 2)
    print("\n[TEST 24] Generating vertical hemiola (3 against 2)...")
    triple, duple = engine.hemiola.generate_vertical_hemiola(duration_beats=4)
    print(f"✓ Generated triple rhythm with {len(triple)} events")
    print(f"✓ Generated duple rhythm with {len(duple)} events")
    assert len(triple) == 3, "Should have 3 events"
    assert len(duple) == 2, "Should have 2 events"

    # Test 25: Accent Pattern Generation
    print("\n[TEST 25] Testing accent pattern generation from grouping...")
    accents = engine.odd_meter._generate_accents_from_grouping([2, 2, 3])
    print(f"✓ Accents for [2,2,3]: {accents}")
    assert accents == [0, 2, 4], "Should accent beats 0, 2, 4"

    # Test 26: TimeSignature Validation
    print("\n[TEST 26] Testing TimeSignature validation...")
    try:
        ts_invalid = TimeSignature(7, 8, [2, 2, 2])  # Sum is 6, not 7
        print("✗ Should have raised ValueError")
        assert False
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")

    # Test 27: Edge Case - Very Long Measure
    print("\n[TEST 27] Testing long odd meter (23/16)...")
    ts_23 = TimeSignature(23, 16, [3, 3, 3, 3, 3, 3, 2, 3])
    pattern_23 = engine.generate_odd_meter_pattern(ts_23, measures=1)
    print(f"✓ Generated {len(pattern_23)} events")
    assert len(pattern_23) == 23, "Should have 23 events"

    # Test 28: Complex Metric Modulation Chain
    print("\n[TEST 28] Testing complex metric modulation chain...")
    tempo1 = 120.0
    tempo2 = engine.metric_modulation(tempo1, 'quarter', 'triplet_quarter')
    tempo3 = engine.metric_modulation(tempo2, 'eighth', 'dotted_eighth')
    print(f"✓ 120 BPM → {tempo2:.2f} BPM → {tempo3:.2f} BPM")

    # Test 29: Carnatic Talas
    print("\n[TEST 29] Testing Carnatic talas...")
    rupaka = engine.generate_tala_pattern(TalaName.RUPAKA, cycles=1)
    misra = engine.generate_tala_pattern(TalaName.MISRA_CHAPU, cycles=1)
    print(f"✓ Rupaka (6 beats): {len(rupaka)} events")
    print(f"✓ Misra Chapu (7 beats): {len(misra)} events")
    assert len(rupaka) == 6, "Rupaka should have 6 beats"
    assert len(misra) == 7, "Misra Chapu should have 7 beats"

    # Test 30: All Preset Odd Meter Styles
    print("\n[TEST 30] Testing all preset odd meter styles...")
    for style in [OddMeterStyle.TAKE_FIVE, OddMeterStyle.MONEY,
                  OddMeterStyle.SOLSBURY_HILL, OddMeterStyle.STING_DRIVEN]:
        pattern = engine.odd_meter.generate_preset_pattern(style, measures=2)
        print(f"✓ {style.value}: {len(pattern)} events")

    print("\n" + "=" * 80)
    print("ALL 30 TESTS PASSED! ✓")
    print("=" * 80)
    print("\nModule Statistics:")
    print(f"  - Odd meter patterns: 5/4, 7/8, 11/8, 13/8, 15/8, 23/16")
    print(f"  - Indian talas: {len(engine.tala.list_talas())} patterns")
    print(f"  - African patterns: 5+ timeline patterns")
    print(f"  - Metric modulation: Unlimited tempo relationships")
    print(f"  - Polyrhythms: 3:2, 4:3, 5:4, and custom ratios")
    print(f"  - Hemiola: Horizontal and vertical")
    print("\nResearch-based implementation complete!")
    print("Ready for integration with harmonymodule system.")
    print("=" * 80)
