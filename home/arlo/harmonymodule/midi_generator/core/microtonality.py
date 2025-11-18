#!/usr/bin/env python3
"""
Microtonality and World Music Scales
=====================================

Comprehensive implementation of microtonal systems and non-Western musical scales.
Provides tools for working with divisions of the octave beyond 12-TET, just intonation,
and traditional scale systems from Arabic, Indian, Turkish, and Persian music.

Theory Background:
-----------------
Western music uses 12-tone equal temperament (12-TET), dividing the octave into
12 equal semitones. However, many musical traditions use different divisions:

- **Equal Temperaments**: Octave divided into N equal parts
  - 24-TET: Quarter tones (Arabic, contemporary classical)
  - 19-TET: Approximates just intonation, 1/3-comma meantone
  - 31-TET: Very close to 1/4-comma meantone
  - 53-TET: Closely approximates just intervals

- **Just Intonation**: Intervals based on pure frequency ratios (3:2, 5:4, etc.)
  Natural acoustic relationships, no equal temperament

- **Non-Western Systems**:
  - Arabic maqam: 24 maqamat using quarter tones
  - Indian raga: 72 melakarta (parent) ragas
  - Turkish makam: 53-TET approximation (Pythagorean comma)
  - Persian dastgah: Seven modal systems with microtones

MIDI Implementation:
-------------------
MIDI pitch bend is used to achieve microtonal intervals:
- Pitch bend range typically set to ±2 semitones
- Resolution: 8192 steps per semitone
- Cents calculation: bend_value = (cents / 100) * 4096 / pitch_bend_range

Applications:
------------
- Contemporary classical music (Partch, Johnston, Wyschnegradsky)
- World music synthesis
- Film scoring (ethnic instruments)
- Experimental electronic music
- Authentic recreation of non-Western music

References:
----------
- Harry Partch: "Genesis of a Music" (just intonation)
- Habib Hassan Touma: "The Music of the Arabs" (maqam system)
- N. A. Jairazbhoy: "The Rāgs of North Indian Music" (raga theory)
- Arel-Ezgi-Uzdilek: Turkish makam theory
- Hormoz Farhat: "The Dastgāh Concept in Persian Music"

Author: Agent 3 - Advanced Harmony & Modal Systems
License: MIT
"""

from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math


# ============================================================================
# CORE MICROTONAL STRUCTURES
# ============================================================================

@dataclass
class MicrotonalInterval:
    """
    Microtonal interval representation.

    Attributes:
        cents: Interval size in cents (1200 cents = octave)
        ratio: Optional frequency ratio (for just intonation)
        name: Interval name
    """
    cents: float
    ratio: Optional[Tuple[int, int]] = None
    name: str = ""

    def to_midi_pitch_bend(self, base_note: int, bend_range: int = 2) -> Tuple[int, int]:
        """
        Convert to MIDI note + pitch bend.

        Args:
            base_note: Base MIDI note number
            bend_range: Pitch bend range in semitones (typically 2)

        Returns:
            Tuple of (MIDI note, pitch bend value -8192 to 8191)
        """
        # Calculate how many semitones up from base note
        semitones = self.cents / 100.0

        # Find closest MIDI note
        midi_note = base_note + round(semitones)

        # Calculate remaining cents to bend
        note_offset = (midi_note - base_note) * 100
        bend_cents = self.cents - note_offset

        # Convert to pitch bend value
        # Pitch bend range: -8192 to 8191 (14-bit)
        # Each unit = (bend_range * 100) / 8192 cents
        bend_value = int((bend_cents * 8192) / (bend_range * 100))
        bend_value = max(-8192, min(8191, bend_value))

        return (midi_note, bend_value)

    @classmethod
    def from_ratio(cls, ratio: Tuple[int, int], name: str = "") -> 'MicrotonalInterval':
        """
        Create interval from frequency ratio.

        Args:
            ratio: Frequency ratio (numerator, denominator)
            name: Interval name

        Returns:
            MicrotonalInterval instance
        """
        cents = 1200 * math.log2(ratio[0] / ratio[1])
        return cls(cents=cents, ratio=ratio, name=name)


@dataclass
class MicrotonalScale:
    """
    Microtonal scale definition.

    Attributes:
        name: Scale name
        intervals: List of intervals in cents from tonic
        tuning_system: Description of tuning system
        culture: Cultural origin (Arabic, Indian, etc.)
    """
    name: str
    intervals: List[float]
    tuning_system: str
    culture: str = "Western"

    def get_pitches(self, root_midi: int, octave_span: int = 1,
                   bend_range: int = 2) -> List[Tuple[int, int]]:
        """
        Get MIDI notes with pitch bends for scale.

        Args:
            root_midi: Root MIDI note
            octave_span: Number of octaves to generate
            bend_range: Pitch bend range

        Returns:
            List of (MIDI note, pitch bend) tuples
        """
        pitches = []
        for octave in range(octave_span):
            for cents in self.intervals:
                total_cents = cents + (octave * 1200)
                interval = MicrotonalInterval(total_cents)
                note, bend = interval.to_midi_pitch_bend(root_midi, bend_range)
                pitches.append((note, bend))

        return pitches


# ============================================================================
# EQUAL TEMPERAMENT SYSTEMS
# ============================================================================

class EqualTemperament:
    """
    Equal division of the octave (EDO/TET).

    Generates scales for N-tone equal temperament.
    """

    def __init__(self, divisions: int):
        """
        Initialize equal temperament system.

        Args:
            divisions: Number of equal divisions per octave
        """
        self.divisions = divisions
        self.step_size = 1200.0 / divisions  # Cents per step

    def get_scale(self, steps: List[int], name: str = "") -> MicrotonalScale:
        """
        Generate scale from step pattern.

        Args:
            steps: List of step numbers (0 = root, divisions = octave)
            name: Scale name

        Returns:
            MicrotonalScale instance
        """
        if not name:
            name = f"{self.divisions}-TET Scale"

        intervals = [step * self.step_size for step in steps]
        return MicrotonalScale(
            name=name,
            intervals=intervals,
            tuning_system=f"{self.divisions}-TET",
            culture="Western"
        )

    def get_chromatic_scale(self) -> MicrotonalScale:
        """Get full chromatic scale for this temperament"""
        steps = list(range(self.divisions))
        return self.get_scale(steps, f"{self.divisions}-TET Chromatic")


# Common equal temperaments
class CommonET:
    """Common equal temperament systems"""

    @staticmethod
    def get_24tet() -> EqualTemperament:
        """24-TET: Quarter-tone system (Arabic music, contemporary classical)"""
        return EqualTemperament(24)

    @staticmethod
    def get_19tet() -> EqualTemperament:
        """19-TET: Approximates 1/3-comma meantone"""
        return EqualTemperament(19)

    @staticmethod
    def get_31tet() -> EqualTemperament:
        """31-TET: Very close to 1/4-comma meantone"""
        return EqualTemperament(31)

    @staticmethod
    def get_53tet() -> EqualTemperament:
        """53-TET: Turkish music, Pythagorean comma"""
        return EqualTemperament(53)

    @staticmethod
    def get_17tet() -> EqualTemperament:
        """17-TET: Arabic music alternative"""
        return EqualTemperament(17)

    @staticmethod
    def get_22tet() -> EqualTemperament:
        """22-TET: Indian sruti approximation"""
        return EqualTemperament(22)


# ============================================================================
# JUST INTONATION
# ============================================================================

class JustIntonation:
    """
    Just intonation system using pure frequency ratios.

    Based on natural harmonic series and simple integer ratios.
    """

    # Common just intonation intervals
    INTERVALS = {
        "unison": (1, 1),
        "minor_2nd": (16, 15),       # 111.7 cents
        "major_2nd": (9, 8),         # 203.9 cents
        "minor_3rd": (6, 5),         # 315.6 cents
        "major_3rd": (5, 4),         # 386.3 cents
        "perfect_4th": (4, 3),       # 498.0 cents
        "tritone": (45, 32),         # 590.2 cents (augmented 4th)
        "perfect_5th": (3, 2),       # 702.0 cents
        "minor_6th": (8, 5),         # 813.7 cents
        "major_6th": (5, 3),         # 884.4 cents
        "minor_7th": (9, 5),         # 1017.6 cents
        "major_7th": (15, 8),        # 1088.3 cents
        "octave": (2, 1),            # 1200.0 cents
    }

    @classmethod
    def get_interval(cls, name: str) -> MicrotonalInterval:
        """Get just intonation interval by name"""
        ratio = cls.INTERVALS[name]
        return MicrotonalInterval.from_ratio(ratio, name)

    @classmethod
    def build_scale(cls, ratios: List[Tuple[int, int]], name: str = "Just Scale") -> MicrotonalScale:
        """
        Build scale from frequency ratios.

        Args:
            ratios: List of frequency ratios
            name: Scale name

        Returns:
            MicrotonalScale instance
        """
        intervals = []
        for ratio in ratios:
            cents = 1200 * math.log2(ratio[0] / ratio[1])
            intervals.append(cents)

        return MicrotonalScale(
            name=name,
            intervals=intervals,
            tuning_system="Just Intonation",
            culture="Western"
        )

    @classmethod
    def get_major_scale(cls) -> MicrotonalScale:
        """Get just intonation major scale"""
        ratios = [
            (1, 1),    # 1 - unison
            (9, 8),    # 2 - major 2nd
            (5, 4),    # 3 - major 3rd
            (4, 3),    # 4 - perfect 4th
            (3, 2),    # 5 - perfect 5th
            (5, 3),    # 6 - major 6th
            (15, 8),   # 7 - major 7th
            (2, 1),    # 8 - octave
        ]
        return cls.build_scale(ratios, "Just Major Scale")

    @classmethod
    def get_harmonic_series(cls, fundamental_harmonics: int = 16) -> MicrotonalScale:
        """
        Get scale from harmonic series.

        Args:
            fundamental_harmonics: Number of harmonics to include

        Returns:
            MicrotonalScale based on harmonic series
        """
        ratios = [(n, 1) for n in range(1, fundamental_harmonics + 1)]
        return cls.build_scale(ratios, f"Harmonic Series (1-{fundamental_harmonics})")


# ============================================================================
# ARABIC MAQAM SYSTEM
# ============================================================================

class ArabicMaqam(Enum):
    """24 principal Arabic maqamat"""
    RAST = "Rast"
    BAYATI = "Bayati"
    SABA = "Saba"
    HIJAZ = "Hijaz"
    SIKAH = "Sikah"
    NAHAWAND = "Nahawand"
    AJAM = "Ajam"
    KURD = "Kurd"
    NAWA_ATHAR = "Nawa Athar"
    FARAHFAZA = "Farahfaza"
    SAZKAR = "Sazkar"
    BASTANIKAR = "Bastanikar"


class MaqamSystem:
    """
    Arabic maqam system using quarter tones (24-TET approximation).

    The maqam system uses ajnas (plural of jins) - tetrachords and pentachords
    that combine to form complete maqamat.
    """

    # Quarter-tone intervals in cents (24-TET)
    # 0 = unison, 1 = quarter-tone (~50 cents), 2 = semitone (~100 cents), etc.
    QUARTER_TONE = 50.0

    # Common ajnas (building blocks)
    AJNAS = {
        # Tetrachords (4 notes)
        "rast": [0, 4, 7, 10],        # T T ST (whole, whole, half)
        "bayati": [0, 3, 6, 10],      # 3QT 3QT T
        "sikah": [0, 3, 7, 10],       # 3QT T 3QT
        "hijaz": [0, 2, 6, 10],       # ST 3ST ST (augmented 2nd)
        "nahawand": [0, 4, 6, 10],    # T ST T
        "ajam": [0, 4, 8, 10],        # T T ST (major tetrachord)
        "kurd": [0, 2, 6, 10],        # ST T T
        "saba": [0, 3, 6, 8],         # 3QT 3QT QT
        "jiharkah": [0, 4, 8, 11],    # T T 3QT
    }

    # Complete maqamat (combining ajnas)
    MAQAMAT = {
        ArabicMaqam.RAST: {
            "lower": "rast",
            "upper": "rast",
            "description": "Major-like, joyful character",
            "intervals": [0, 4, 7, 10, 14, 17, 21, 24]  # Two rast tetrachords
        },
        ArabicMaqam.BAYATI: {
            "lower": "bayati",
            "upper": "ajam",
            "description": "Minor-like, common in folk music",
            "intervals": [0, 3, 6, 10, 14, 16, 20, 24]
        },
        ArabicMaqam.SABA: {
            "lower": "saba",
            "upper": "hijaz",
            "description": "Melancholic, complex character",
            "intervals": [0, 3, 6, 8, 12, 16, 18, 22]
        },
        ArabicMaqam.HIJAZ: {
            "lower": "hijaz",
            "upper": "rast",
            "description": "Dramatic, augmented 2nd",
            "intervals": [0, 2, 6, 10, 14, 17, 20, 24]
        },
        ArabicMaqam.SIKAH: {
            "lower": "sikah",
            "upper": "sikah",
            "description": "Medium register, distinctive 3QT start",
            "intervals": [0, 3, 7, 10, 14, 17, 21, 24]
        },
        ArabicMaqam.NAHAWAND: {
            "lower": "nahawand",
            "upper": "nahawand",
            "description": "Natural minor-like",
            "intervals": [0, 4, 6, 10, 14, 16, 20, 24]
        },
        ArabicMaqam.AJAM: {
            "lower": "ajam",
            "upper": "ajam",
            "description": "Western major scale equivalent",
            "intervals": [0, 4, 8, 10, 14, 18, 22, 24]
        },
        ArabicMaqam.KURD: {
            "lower": "kurd",
            "upper": "nahawand",
            "description": "Minor-like, common in Turkish music",
            "intervals": [0, 2, 6, 10, 14, 16, 20, 24]
        },
    }

    @classmethod
    def get_maqam(cls, maqam: ArabicMaqam) -> MicrotonalScale:
        """
        Get maqam scale.

        Args:
            maqam: Maqam to retrieve

        Returns:
            MicrotonalScale instance
        """
        maqam_def = cls.MAQAMAT[maqam]
        intervals_cents = [step * cls.QUARTER_TONE for step in maqam_def["intervals"]]

        return MicrotonalScale(
            name=f"Maqam {maqam.value}",
            intervals=intervals_cents,
            tuning_system="24-TET (Quarter-tone)",
            culture="Arabic"
        )

    @classmethod
    def get_jins(cls, jins_name: str) -> List[float]:
        """
        Get jins (tetrachord/pentachord) in cents.

        Args:
            jins_name: Name of jins

        Returns:
            List of intervals in cents
        """
        steps = cls.AJNAS[jins_name]
        return [step * cls.QUARTER_TONE for step in steps]


# ============================================================================
# INDIAN RAGA SYSTEM
# ============================================================================

class IndianRaga:
    """
    Indian raga system based on 72 melakarta (parent) ragas.

    Uses 12 svaras (notes) with two variants for each (except Sa and Pa).
    Approximated using 12-TET with some ragas requiring 22-sruti approximation.
    """

    # The 12 svaras (note names)
    SVARAS = ["Sa", "Ri", "Ga", "Ma", "Pa", "Dha", "Ni"]

    # Melakarta system: 72 parent ragas
    # Each has specific intervallic structure
    # Example melas (simplified - normally requires 22-sruti system)

    COMMON_RAGAS = {
        "Bhairav": {
            "intervals": [0, 1, 4, 5, 7, 8, 11, 12],  # Approximation in 12-TET
            "arohana": [0, 1, 4, 5, 7, 8, 11, 12],    # Ascending
            "avarohana": [12, 11, 8, 7, 5, 4, 1, 0],  # Descending
            "vadi": 4,    # Most important note (Ma)
            "samvadi": 8, # Second most important (Dha)
            "time": "morning",
            "rasa": "devotional"
        },
        "Yaman": {
            "intervals": [0, 2, 4, 6, 7, 9, 11, 12],
            "arohana": [0, 2, 4, 6, 7, 9, 11, 12],
            "avarohana": [12, 11, 9, 7, 6, 4, 2, 0],
            "vadi": 7,
            "samvadi": 2,
            "time": "evening",
            "rasa": "romantic"
        },
        "Kafi": {
            "intervals": [0, 2, 3, 5, 7, 9, 10, 12],
            "arohana": [0, 2, 3, 5, 7, 9, 10, 12],
            "avarohana": [12, 10, 9, 7, 5, 3, 2, 0],
            "vadi": 7,
            "samvadi": 3,
            "time": "night",
            "rasa": "peaceful"
        },
    }

    @classmethod
    def get_raga(cls, raga_name: str) -> Dict:
        """Get raga definition with metadata"""
        return cls.COMMON_RAGAS.get(raga_name, {})

    @classmethod
    def get_raga_scale(cls, raga_name: str, use_arohana: bool = True) -> MicrotonalScale:
        """
        Get raga scale (ascending or descending).

        Args:
            raga_name: Name of raga
            use_arohana: True for ascending, False for descending

        Returns:
            MicrotonalScale instance
        """
        raga = cls.get_raga(raga_name)
        intervals_semitones = raga["arohana"] if use_arohana else raga["avarohana"]
        intervals_cents = [st * 100 for st in intervals_semitones]

        return MicrotonalScale(
            name=f"Raga {raga_name}",
            intervals=intervals_cents,
            tuning_system="12-TET (Approximation)",
            culture="Indian"
        )


# ============================================================================
# TURKISH MAKAM SYSTEM
# ============================================================================

class TurkishMakam:
    """
    Turkish makam system using 53-TET (Pythagorean comma approximation).

    Turkish music theory divides the octave into 53 equal parts (Holdrian comma).
    """

    # 53-TET step = ~22.64 cents
    COMMA = 1200.0 / 53.0  # Holdrian comma

    # Common makamlar (plural of makam)
    MAKAMLAR = {
        "Hicaz": {
            "intervals": [0, 5, 17, 22, 31, 36, 48, 53],
            "description": "Similar to Phrygian dominant"
        },
        "Rast": {
            "intervals": [0, 9, 18, 22, 31, 40, 49, 53],
            "description": "Major-like"
        },
        "Hüseyni": {
            "intervals": [0, 9, 13, 22, 31, 40, 44, 53],
            "description": "Minor-like"
        },
        "Kürdi": {
            "intervals": [0, 4, 13, 22, 31, 35, 44, 53],
            "description": "Minor with quarter tones"
        },
        "Segah": {
            "intervals": [0, 9, 13, 22, 31, 36, 44, 53],
            "description": "Medium register characteristic"
        },
    }

    @classmethod
    def get_makam(cls, makam_name: str) -> MicrotonalScale:
        """
        Get Turkish makam scale.

        Args:
            makam_name: Name of makam

        Returns:
            MicrotonalScale instance
        """
        makam_def = cls.MAKAMLAR[makam_name]
        intervals_cents = [step * cls.COMMA for step in makam_def["intervals"]]

        return MicrotonalScale(
            name=f"Makam {makam_name}",
            intervals=intervals_cents,
            tuning_system="53-TET",
            culture="Turkish"
        )


# ============================================================================
# PERSIAN DASTGAH SYSTEM
# ============================================================================

class PersianDastgah:
    """
    Persian dastgah system with seven principal dastgah-ha.

    Uses intervals that include neutral intervals (between major and minor).
    Approximated using 24-TET.
    """

    DASTGAH_HA = {
        "Shur": {
            "intervals": [0, 4, 5, 10, 14, 17, 19, 24],
            "gooshe_count": 12,  # Number of gooshe-ha (melodic motifs)
            "description": "Most common, melancholic"
        },
        "Mahur": {
            "intervals": [0, 4, 8, 10, 14, 18, 22, 24],
            "gooshe_count": 8,
            "description": "Major-like, joyful"
        },
        "Homayoun": {
            "intervals": [0, 2, 6, 10, 14, 16, 20, 24],
            "gooshe_count": 10,
            "description": "Minor-like, contemplative"
        },
        "Segah": {
            "intervals": [0, 3, 7, 10, 14, 17, 21, 24],
            "gooshe_count": 9,
            "description": "Neutral intervals, morning music"
        },
    }

    @classmethod
    def get_dastgah(cls, dastgah_name: str) -> MicrotonalScale:
        """
        Get Persian dastgah scale.

        Args:
            dastgah_name: Name of dastgah

        Returns:
            MicrotonalScale instance
        """
        dastgah_def = cls.DASTGAH_HA[dastgah_name]
        # Using 24-TET approximation (quarter tones)
        quarter_tone = 50.0
        intervals_cents = [step * quarter_tone for step in dastgah_def["intervals"]]

        return MicrotonalScale(
            name=f"Dastgah {dastgah_name}",
            intervals=intervals_cents,
            tuning_system="24-TET (Quarter-tone approximation)",
            culture="Persian"
        )


# ============================================================================
# MIDI IMPLEMENTATION HELPERS
# ============================================================================

class MicrotonalMIDI:
    """
    Helper class for implementing microtonal music in MIDI.

    Provides utilities for pitch bend calculations and multi-channel strategies.
    """

    @staticmethod
    def calculate_pitch_bend(cents: float, bend_range: int = 2) -> int:
        """
        Calculate MIDI pitch bend value from cents.

        Args:
            cents: Cents deviation from base note
            bend_range: Pitch bend range in semitones

        Returns:
            Pitch bend value (-8192 to 8191)
        """
        # Pitch bend: -8192 to 8191 (14-bit, 0 = center)
        cents_per_unit = (bend_range * 100.0) / 8192.0
        bend_value = int(cents / cents_per_unit)
        return max(-8192, min(8191, bend_value))

    @staticmethod
    def create_pitch_bend_message(channel: int, bend_value: int) -> Tuple[int, int, int]:
        """
        Create MIDI pitch bend message.

        Args:
            channel: MIDI channel (0-15)
            bend_value: Pitch bend value (-8192 to 8191)

        Returns:
            Tuple of (status, lsb, msb)
        """
        # Convert to 0-16383 range
        bend_14bit = bend_value + 8192

        # Split into LSB and MSB (7-bit each)
        lsb = bend_14bit & 0x7F
        msb = (bend_14bit >> 7) & 0x7F

        status = 0xE0 | channel  # Pitch bend status + channel

        return (status, lsb, msb)

    @staticmethod
    def microtonal_note_on(note: int, cents_offset: float, velocity: int = 64,
                          channel: int = 0, bend_range: int = 2) -> List[Tuple]:
        """
        Generate MIDI messages for microtonal note.

        Args:
            note: Base MIDI note
            cents_offset: Cents offset from base note
            velocity: Note velocity
            channel: MIDI channel
            bend_range: Pitch bend range

        Returns:
            List of MIDI message tuples
        """
        messages = []

        # Set pitch bend
        bend_value = MicrotonalMIDI.calculate_pitch_bend(cents_offset, bend_range)
        bend_msg = MicrotonalMIDI.create_pitch_bend_message(channel, bend_value)
        messages.append(("pitch_bend", bend_msg))

        # Note on
        note_on = (0x90 | channel, note, velocity)
        messages.append(("note_on", note_on))

        return messages


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MICROTONALITY & WORLD MUSIC SCALES - EXAMPLES")
    print("=" * 70)

    # Example 1: 24-TET quarter-tone scale
    print("\n1. 24-TET QUARTER-TONE SYSTEM")
    print("-" * 70)
    tet24 = CommonET.get_24tet()
    chromatic_24 = tet24.get_chromatic_scale()
    print(f"{chromatic_24.name}:")
    print(f"  Steps per octave: {len(chromatic_24.intervals)}")
    print(f"  Step size: {tet24.step_size:.2f} cents")
    print(f"  First 5 intervals: {chromatic_24.intervals[:5]}")

    # Example 2: Just intonation major scale
    print("\n2. JUST INTONATION MAJOR SCALE")
    print("-" * 70)
    just_major = JustIntonation.get_major_scale()
    print(f"{just_major.name}:")
    for i, cents in enumerate(just_major.intervals):
        degree = i + 1
        print(f"  Degree {degree}: {cents:.2f} cents")

    # Example 3: Arabic Maqam Rast
    print("\n3. ARABIC MAQAM RAST")
    print("-" * 70)
    maqam_rast = MaqamSystem.get_maqam(ArabicMaqam.RAST)
    print(f"{maqam_rast.name}:")
    print(f"  Culture: {maqam_rast.culture}")
    print(f"  Tuning: {maqam_rast.tuning_system}")
    print(f"  Intervals (cents): {[f'{c:.1f}' for c in maqam_rast.intervals]}")

    # Example 4: Arabic Maqam Hijaz
    print("\n4. ARABIC MAQAM HIJAZ")
    print("-" * 70)
    maqam_hijaz = MaqamSystem.get_maqam(ArabicMaqam.HIJAZ)
    print(f"{maqam_hijaz.name}:")
    print(f"  Intervals (cents): {[f'{c:.1f}' for c in maqam_hijaz.intervals]}")
    print(f"  Characteristic: Augmented 2nd interval")

    # Example 5: Indian Raga Bhairav
    print("\n5. INDIAN RAGA BHAIRAV")
    print("-" * 70)
    bhairav = IndianRaga.get_raga_scale("Bhairav")
    raga_info = IndianRaga.get_raga("Bhairav")
    print(f"{bhairav.name}:")
    print(f"  Time: {raga_info['time']}")
    print(f"  Rasa (mood): {raga_info['rasa']}")
    print(f"  Vadi (main note): Scale degree {raga_info['vadi']}")
    print(f"  Intervals: {bhairav.intervals}")

    # Example 6: Turkish Makam Hicaz
    print("\n6. TURKISH MAKAM HICAZ")
    print("-" * 70)
    makam_hicaz = TurkishMakam.get_makam("Hicaz")
    print(f"{makam_hicaz.name}:")
    print(f"  Tuning: {makam_hicaz.tuning_system}")
    print(f"  Intervals (cents): {[f'{c:.1f}' for c in makam_hicaz.intervals]}")

    # Example 7: Persian Dastgah Shur
    print("\n7. PERSIAN DASTGAH SHUR")
    print("-" * 70)
    dastgah_shur = PersianDastgah.get_dastgah("Shur")
    dastgah_info = PersianDastgah.DASTGAH_HA["Shur"]
    print(f"{dastgah_shur.name}:")
    print(f"  Description: {dastgah_info['description']}")
    print(f"  Number of gooshe: {dastgah_info['gooshe_count']}")
    print(f"  Intervals (cents): {[f'{c:.1f}' for c in dastgah_shur.intervals]}")

    # Example 8: MIDI pitch bend calculation
    print("\n8. MIDI PITCH BEND FOR MICROTONES")
    print("-" * 70)
    # Quarter-tone above C4 (60)
    quarter_tone_cents = 50.0
    bend_value = MicrotonalMIDI.calculate_pitch_bend(quarter_tone_cents, bend_range=2)
    print(f"Quarter-tone (50 cents):")
    print(f"  Base note: C4 (MIDI 60)")
    print(f"  Pitch bend value: {bend_value}")
    print(f"  Pitch bend message: {MicrotonalMIDI.create_pitch_bend_message(0, bend_value)}")

    # Example 9: 53-TET (Turkish system)
    print("\n9. 53-TET SYSTEM")
    print("-" * 70)
    tet53 = CommonET.get_53tet()
    print(f"53-TET (Turkish/Pythagorean):")
    print(f"  Step size: {tet53.step_size:.2f} cents (Holdrian comma)")
    print(f"  Steps per octave: 53")
    print(f"  Use: Turkish makam music")

    # Example 10: Harmonic series scale
    print("\n10. HARMONIC SERIES SCALE")
    print("-" * 70)
    harmonic_scale = JustIntonation.get_harmonic_series(8)
    print(f"{harmonic_scale.name}:")
    print(f"  Intervals (cents):")
    for i, cents in enumerate(harmonic_scale.intervals[:8], 1):
        print(f"    Harmonic {i}: {cents:.2f} cents")

    print("\n" + "=" * 70)
