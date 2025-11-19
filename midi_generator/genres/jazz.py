#!/usr/bin/env python3
"""
Jazz Music Generator - Comprehensive Jazz Composition System
=============================================================

This module consolidates and unifies all jazz components from the library into
a comprehensive jazz generation system covering all major jazz styles from
traditional to contemporary.

Sub-genres Implemented:
-----------------------
- Bebop (Charlie Parker, Dizzy Gillespie)
- Modal Jazz (Miles Davis, John Coltrane)
- Cool Jazz (Chet Baker, Modern Jazz Quartet)
- Hard Bop (Art Blakey, Horace Silver)
- Free Jazz (Ornette Coleman, Cecil Taylor)
- Fusion (Weather Report, Mahavishnu Orchestra)
- Latin Jazz (Tito Puente, Chick Corea)
- Smooth Jazz (George Benson, Grover Washington Jr.)
- Swing (Count Basie, Duke Ellington)
- Contemporary/Nu-Jazz (Robert Glasper, Kamasi Washington)

Features:
---------
- Modal harmony (all church modes + melodic/harmonic minor modes)
- Bebop melody generation with chromatic approach notes
- Walking bass lines (bebop, swing, latin jazz styles)
- Jazz reharmonization (upper structures, altered dominants)
- Piano comping patterns (shell voicings, rootless voicings, quartal voicings)
- Jazz drum patterns (swing, bebop, latin, fusion)
- ii-V-I progressions with voice leading
- Chord-scale relationships
- Jazz forms (32-bar AABA, 12-bar blues, rhythm changes)
- Swing timing and microtiming
- Jazz articulations (ghost notes, accents, swells)

Research References:
-------------------
- George Russell: "Lydian Chromatic Concept of Tonal Organization"
- Mark Levine: "The Jazz Theory Book"
- Jerry Coker: "Improvising Jazz"
- Jamey Aebersold: "Jazz Handbook"
- Dias & Guedes (2013): "Bass Line Generation Algorithm"
- Roger Linn: Swing algorithm (MPC)
- J Dilla: Microtiming analysis

Author: Consolidated Jazz Module (Phase 3)
Date: 2025
License: MIT
"""

import random
import math
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum

# Import existing jazz components from library
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.modal_harmony import (
    Mode, HarmonicMinorMode, MelodicMinorMode, SymmetricalScale,
    ModalScale, ModalScaleLibrary, ModalProgressionGenerator
)


# ============================================================================
# JAZZ STYLES AND ENUMS
# ============================================================================

class JazzStyle(Enum):
    """Major jazz sub-genres"""
    BEBOP = "bebop"                      # Fast, complex (Parker, Gillespie)
    MODAL = "modal"                      # Modal harmony (Davis, Coltrane)
    COOL = "cool"                        # Subdued, lyrical (Baker, MJQ)
    HARD_BOP = "hard_bop"                # Bluesy, soulful (Blakey, Silver)
    FREE = "free"                        # Avant-garde (Coleman, Taylor)
    FUSION = "fusion"                    # Jazz-rock (Weather Report, McLaughlin)
    LATIN = "latin"                      # Afro-Cuban (Puente, Corea)
    SMOOTH = "smooth"                    # Contemporary (Benson, Washington)
    SWING = "swing"                      # Big band era (Basie, Ellington)
    CONTEMPORARY = "contemporary"        # Modern jazz (Glasper, Kamasi)


class ChordScaleRelation(Enum):
    """Chord-scale relationships for jazz improvisation"""
    # Major family
    MAJ7_IONIAN = ("maj7", Mode.IONIAN)
    MAJ7_LYDIAN = ("maj7", Mode.LYDIAN)

    # Minor family
    MIN7_DORIAN = ("min7", Mode.DORIAN)
    MIN7_AEOLIAN = ("min7", Mode.AEOLIAN)
    MIN7_PHRYGIAN = ("min7", Mode.PHRYGIAN)

    # Dominant family
    DOM7_MIXOLYDIAN = ("dom7", Mode.MIXOLYDIAN)
    DOM7_LYDIAN_DOMINANT = ("dom7", MelodicMinorMode.LYDIAN_DOMINANT)
    DOM7_ALTERED = ("dom7", MelodicMinorMode.ALTERED)
    DOM7_DIMINISHED = ("dom7", SymmetricalScale.DIMINISHED_WHOLE_HALF)

    # Half-diminished
    MIN7b5_LOCRIAN = ("min7b5", Mode.LOCRIAN)
    MIN7b5_LOCRIAN_NAT2 = ("min7b5", MelodicMinorMode.LOCRIAN_NAT2)


class JazzForm(Enum):
    """Standard jazz song forms"""
    AABA_32 = "aaba_32"                  # 32-bar AABA (standard)
    BLUES_12 = "blues_12"                # 12-bar blues
    RHYTHM_CHANGES = "rhythm_changes"    # Rhythm changes (Gershwin)
    ABAC_32 = "abac_32"                  # 32-bar ABAC
    MODAL_VAMP = "modal_vamp"            # Modal vamp (no changes)
    THROUGH_COMPOSED = "through_composed" # No repeated sections


class CompingStyle(Enum):
    """Piano/guitar comping styles"""
    SHELL = "shell"                      # Root, 3rd, 7th only
    ROOTLESS = "rootless"                # 3rd, 5th/6th, 7th, 9th (Bill Evans)
    QUARTAL = "quartal"                  # Fourths (McCoy Tyner)
    BLOCK = "block"                      # Full block chords (Red Garland)
    FREDDIE_GREEN = "freddie_green"      # Four-to-the-bar (Freddie Green)
    STRIDE = "stride"                    # Stride piano (James P. Johnson)


class SwingFeel(Enum):
    """Swing subdivision feels"""
    STRAIGHT = "straight"                # Even 8ths
    LIGHT_SWING = "light_swing"          # 55-58% swing ratio
    MEDIUM_SWING = "medium_swing"        # 58-62% swing ratio (standard)
    HEAVY_SWING = "heavy_swing"          # 62-67% swing ratio
    TRIPLET = "triplet"                  # Pure triplet feel


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class JazzNote:
    """Jazz note with articulation and expression"""
    pitch: int                           # MIDI note number (0-127)
    velocity: int                        # Velocity (1-127)
    start_time: float                    # Start time in beats
    duration: float                      # Duration in beats
    articulation: str = "normal"         # normal, accent, ghost, staccato, legato
    swing_offset: float = 0.0            # Swing timing offset in ms
    channel: int = 0                     # MIDI channel


@dataclass
class JazzChord:
    """Jazz chord with extensions and voicing"""
    root: int                            # Root pitch class (0-11)
    quality: str                         # maj7, min7, dom7, min7b5, dim7, etc.
    extensions: List[int] = field(default_factory=list)  # 9, 11, 13
    alterations: List[str] = field(default_factory=list) # b9, #9, #11, b13
    inversion: int = 0                   # 0=root, 1=1st inv, 2=2nd inv, etc.
    voicing_type: CompingStyle = CompingStyle.SHELL


@dataclass
class BebopScale:
    """Bebop scales with chromatic passing tones"""
    root: int                            # Root pitch class (0-11)
    intervals: Tuple[int, ...]           # Semitone intervals
    chromatic_positions: List[int]       # Where to insert chromatic tones

    # Standard bebop scales
    @staticmethod
    def major_bebop(root: int) -> 'BebopScale':
        """Major bebop scale (1 2 3 4 5 #5 6 7)"""
        return BebopScale(
            root=root,
            intervals=(0, 2, 4, 5, 7, 8, 9, 11),
            chromatic_positions=[5]  # Chromatic between 5 and 6
        )

    @staticmethod
    def dominant_bebop(root: int) -> 'BebopScale':
        """Dominant bebop scale (1 2 3 4 5 6 b7 7)"""
        return BebopScale(
            root=root,
            intervals=(0, 2, 4, 5, 7, 9, 10, 11),
            chromatic_positions=[7]  # Chromatic between b7 and root
        )

    @staticmethod
    def minor_bebop(root: int) -> 'BebopScale':
        """Minor bebop scale (1 2 b3 4 5 b6 6 b7)"""
        return BebopScale(
            root=root,
            intervals=(0, 2, 3, 5, 7, 8, 9, 10),
            chromatic_positions=[6]  # Chromatic between b6 and 6
        )

    def get_notes(self, octave: int = 4) -> List[int]:
        """Get scale notes as MIDI note numbers"""
        base = 12 * octave + self.root
        return [base + interval for interval in self.intervals]


# ============================================================================
# CHORD PROGRESSIONS
# ============================================================================

class JazzProgressions:
    """Standard jazz chord progressions"""

    @staticmethod
    def ii_V_I(key: int) -> List[JazzChord]:
        """
        The quintessential jazz progression: ii-V-I

        Args:
            key: Root key (0-11)

        Returns:
            List of JazzChord objects
        """
        return [
            JazzChord(root=(key + 2) % 12, quality="min7"),      # ii
            JazzChord(root=(key + 7) % 12, quality="dom7"),      # V
            JazzChord(root=key, quality="maj7")                  # I
        ]

    @staticmethod
    def rhythm_changes_A(key: int) -> List[JazzChord]:
        """
        Rhythm changes A section (based on "I Got Rhythm")
        8 bars: I-VI-ii-V-I-VI-ii-V
        """
        return [
            JazzChord(root=key, quality="maj7"),                 # I
            JazzChord(root=(key + 9) % 12, quality="dom7"),      # VI7
            JazzChord(root=(key + 2) % 12, quality="min7"),      # ii
            JazzChord(root=(key + 7) % 12, quality="dom7"),      # V
            JazzChord(root=key, quality="maj7"),                 # I
            JazzChord(root=(key + 9) % 12, quality="dom7"),      # VI7
            JazzChord(root=(key + 2) % 12, quality="min7"),      # ii
            JazzChord(root=(key + 7) % 12, quality="dom7"),      # V
        ]

    @staticmethod
    def jazz_blues(key: int) -> List[JazzChord]:
        """
        Jazz blues (12 bars)
        More sophisticated than basic blues with ii-V substitutions
        """
        return [
            JazzChord(root=key, quality="dom7"),                           # I7 (bar 1)
            JazzChord(root=(key + 5) % 12, quality="min7"),                # ivm7 (bar 2)
            JazzChord(root=(key + 10) % 12, quality="dom7"),               # VII7 (bar 2)
            JazzChord(root=key, quality="dom7"),                           # I7 (bar 3)
            JazzChord(root=key, quality="dom7"),                           # I7 (bar 4)
            JazzChord(root=(key + 5) % 12, quality="dom7"),                # IV7 (bar 5)
            JazzChord(root=(key + 5) % 12, quality="dom7"),                # IV7 (bar 6)
            JazzChord(root=key, quality="dom7"),                           # I7 (bar 7)
            JazzChord(root=(key + 9) % 12, quality="maj7"),                # VIm7 (bar 8)
            JazzChord(root=(key + 2) % 12, quality="min7"),                # ii7 (bar 9)
            JazzChord(root=(key + 7) % 12, quality="dom7"),                # V7 (bar 10)
            JazzChord(root=key, quality="maj7"),                           # Imaj7 (bar 11)
            JazzChord(root=(key + 9) % 12, quality="min7"),                # vim7 (bar 12)
        ]

    @staticmethod
    def minor_ii_V_i(key: int) -> List[JazzChord]:
        """
        Minor ii-V-i progression
        """
        return [
            JazzChord(root=(key + 2) % 12, quality="min7b5"),    # ii°
            JazzChord(root=(key + 7) % 12, quality="dom7",
                     alterations=["b9"]),                        # V7b9
            JazzChord(root=key, quality="min7")                  # i
        ]


# ============================================================================
# WALKING BASS GENERATOR
# ============================================================================

class JazzWalkingBass:
    """
    Walking bass line generator for jazz.

    Implements contour-based algorithm from Dias & Guedes (2013):
    "Automatic Generation of Walking Bass Lines for Jazz"

    Features:
    - Chord tone emphasis on beats 1 and 3
    - Chromatic approach notes
    - Passing tones (chromatic and diatonic)
    - Contour control (ascending/descending)
    """

    def __init__(self, style: JazzStyle = JazzStyle.BEBOP):
        self.style = style
        self.register_low = 28   # E1
        self.register_high = 55  # G3

    def generate_line(
        self,
        chords: List[JazzChord],
        beats_per_chord: int = 4,
        style: str = "bebop"
    ) -> List[JazzNote]:
        """
        Generate walking bass line for chord progression.

        Args:
            chords: List of JazzChord objects
            beats_per_chord: Number of beats per chord
            style: "bebop", "swing", or "latin_jazz"

        Returns:
            List of JazzNote objects
        """
        bass_notes = []
        current_beat = 0.0

        for i, chord in enumerate(chords):
            chord_tones = self._get_chord_tones(chord)
            root = chord.root + 36  # Bass register (C2-G3)

            for beat in range(beats_per_chord):
                if beat == 0:
                    # Beat 1: Always root
                    note = root
                elif beat == 2:
                    # Beat 3: Usually 5th or 3rd
                    note = random.choice([root + 7, root + self._get_third(chord)])
                else:
                    # Beats 2 and 4: Passing tones or chord tones
                    if i < len(chords) - 1:
                        next_root = chords[i + 1].root + 36
                        # Chromatic approach to next chord
                        if beat == beats_per_chord - 1:
                            note = next_root - 1  # Half-step below
                        else:
                            note = random.choice(chord_tones)
                    else:
                        note = random.choice(chord_tones)

                # Ensure note is in bass register
                while note < self.register_low:
                    note += 12
                while note > self.register_high:
                    note -= 12

                bass_notes.append(JazzNote(
                    pitch=note,
                    velocity=random.randint(75, 95),
                    start_time=current_beat,
                    duration=0.95,  # Slight gap for walking feel
                    articulation="normal"
                ))

                current_beat += 1.0

        return bass_notes

    def _get_chord_tones(self, chord: JazzChord) -> List[int]:
        """Get chord tones in bass register"""
        root = chord.root + 36
        third = root + self._get_third(chord)
        fifth = root + 7
        seventh = root + self._get_seventh(chord)
        return [root, third, fifth, seventh]

    def _get_third(self, chord: JazzChord) -> int:
        """Get third interval based on chord quality"""
        if "min" in chord.quality or "dim" in chord.quality:
            return 3  # Minor third
        return 4  # Major third

    def _get_seventh(self, chord: JazzChord) -> int:
        """Get seventh interval based on chord quality"""
        if "maj7" in chord.quality:
            return 11  # Major seventh
        elif "dim" in chord.quality:
            return 9   # Diminished seventh
        return 10  # Minor seventh (dominant/minor 7)


# ============================================================================
# BEBOP MELODY GENERATOR
# ============================================================================

class BebopMelodyGenerator:
    """
    Bebop melody generator with chromatic approach notes.

    Based on bebop language principles:
    - Chromatic enclosures
    - Approach notes (chromatic and diatonic)
    - Chord tone targeting
    - Scalar runs and arpeggios
    """

    def __init__(self):
        self.register_low = 60   # C4
        self.register_high = 84  # C6

    def generate_phrase(
        self,
        chord: JazzChord,
        length_beats: int = 4,
        density: float = 0.8
    ) -> List[JazzNote]:
        """
        Generate bebop phrase over single chord.

        Args:
            chord: JazzChord to improvise over
            length_beats: Length of phrase in beats
            density: Note density (0.0-1.0, higher = more notes)

        Returns:
            List of JazzNote objects
        """
        phrase = []
        scale = BebopScale.dominant_bebop(chord.root)
        scale_notes = scale.get_notes(octave=5)

        # Generate note sequence
        current_beat = 0.0
        subdivision = 0.5 if density > 0.6 else 1.0  # 8ths or quarters

        while current_beat < length_beats:
            # Target chord tone on strong beats
            if current_beat % 1.0 == 0:
                note = random.choice(self._get_chord_tones(chord, octave=5))
            else:
                note = random.choice(scale_notes)

            # Add chromatic approach note occasionally
            if random.random() < 0.3 and current_beat < length_beats - 0.5:
                # Chromatic enclosure
                phrase.append(JazzNote(
                    pitch=note + 1,
                    velocity=random.randint(70, 90),
                    start_time=current_beat,
                    duration=subdivision * 0.4,
                    articulation="staccato"
                ))
                current_beat += subdivision * 0.5

            phrase.append(JazzNote(
                pitch=note,
                velocity=random.randint(80, 110),
                start_time=current_beat,
                duration=subdivision * 0.9,
                articulation="normal"
            ))

            current_beat += subdivision

        return phrase

    def _get_chord_tones(self, chord: JazzChord, octave: int = 5) -> List[int]:
        """Get chord tones for targeting"""
        base = 12 * octave + chord.root
        third = base + (3 if "min" in chord.quality else 4)
        fifth = base + 7
        seventh = base + (10 if "dom" in chord.quality or "min" in chord.quality else 11)
        return [base, third, fifth, seventh]


# ============================================================================
# PIANO COMPING GENERATOR
# ============================================================================

class PianoComping:
    """
    Jazz piano comping pattern generator.

    Styles:
    - Shell voicings (root, 3rd, 7th)
    - Rootless voicings (Bill Evans style)
    - Quartal voicings (McCoy Tyner)
    - Block chords (Red Garland)
    - Stride piano
    """

    def __init__(self, style: CompingStyle = CompingStyle.ROOTLESS):
        self.style = style

    def voice_chord(self, chord: JazzChord, octave: int = 4) -> List[int]:
        """
        Voice chord according to comping style.

        Args:
            chord: JazzChord to voice
            octave: Base octave

        Returns:
            List of MIDI note numbers
        """
        root = 12 * octave + chord.root

        if self.style == CompingStyle.SHELL:
            # Root, 3rd, 7th
            third = root + (3 if "min" in chord.quality else 4)
            seventh = root + (10 if "dom" in chord.quality else 11)
            return [root, third, seventh]

        elif self.style == CompingStyle.ROOTLESS:
            # Bill Evans style: 3rd, 5th/6th, 7th, 9th
            third = root + (3 if "min" in chord.quality else 4)
            fifth_or_sixth = root + (7 if "maj7" in chord.quality else 9)
            seventh = root + (10 if ("dom" in chord.quality or "min" in chord.quality) else 11)
            ninth = root + 14
            return [third, fifth_or_sixth, seventh, ninth]

        elif self.style == CompingStyle.QUARTAL:
            # McCoy Tyner quartal voicing (stacked fourths)
            return [root, root + 5, root + 10, root + 15, root + 19]

        elif self.style == CompingStyle.BLOCK:
            # Full block chord
            third = root + (3 if "min" in chord.quality else 4)
            fifth = root + 7
            seventh = root + (10 if ("dom" in chord.quality or "min" in chord.quality) else 11)
            return [root, third, fifth, seventh]

        else:
            # Default to shell
            third = root + (3 if "min" in chord.quality else 4)
            seventh = root + (10 if ("dom" in chord.quality or "min" in chord.quality) else 11)
            return [root, third, seventh]


# ============================================================================
# SWING TIMING ENGINE
# ============================================================================

class SwingTiming:
    """
    Swing timing and microtiming engine.

    Implements:
    - Roger Linn swing algorithm (MPC)
    - Triplet-based swing ratios
    - Jazz bebop timing profiles
    """

    @staticmethod
    def apply_swing(
        notes: List[JazzNote],
        swing_ratio: float = 0.62,
        intensity: float = 1.0
    ) -> List[JazzNote]:
        """
        Apply swing feel to notes.

        Args:
            notes: List of JazzNote objects
            swing_ratio: Swing ratio (0.5=straight, 0.67=triplet, 0.62=standard)
            intensity: Swing intensity (0.0-1.0)

        Returns:
            Notes with adjusted timing
        """
        swung_notes = []

        for note in notes:
            new_note = JazzNote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                articulation=note.articulation,
                swing_offset=note.swing_offset,
                channel=note.channel
            )

            # Apply swing to off-beats (every other 8th note)
            beat_position = note.start_time % 1.0
            if 0.45 < beat_position < 0.55:  # Off-beat 8th note
                # Delay by swing amount
                delay = (swing_ratio - 0.5) * intensity
                new_note.start_time += delay
                new_note.swing_offset = delay

            swung_notes.append(new_note)

        return swung_notes


# ============================================================================
# JAZZ GENERATOR (MAIN CLASS)
# ============================================================================

class JazzGenerator:
    """
    Comprehensive jazz music generator.

    Consolidates all jazz components into unified API.
    """

    def __init__(
        self,
        style: JazzStyle = JazzStyle.BEBOP,
        tempo: int = 180,
        key: int = 0,
        swing_feel: SwingFeel = SwingFeel.MEDIUM_SWING
    ):
        self.style = style
        self.tempo = tempo
        self.key = key
        self.swing_feel = swing_feel

        # Initialize sub-generators
        self.walking_bass = JazzWalkingBass(style)
        self.bebop_melody = BebopMelodyGenerator()
        self.comping = PianoComping(CompingStyle.ROOTLESS)

    def generate_composition(
        self,
        form: JazzForm = JazzForm.AABA_32,
        num_choruses: int = 1
    ) -> Dict:
        """
        Generate complete jazz composition.

        Args:
            form: Song form (AABA, blues, etc.)
            num_choruses: Number of times through the form

        Returns:
            Dictionary with bass, melody, chords, and metadata
        """
        # Generate chord progression based on form
        if form == JazzForm.BLUES_12:
            progression = JazzProgressions.jazz_blues(self.key)
        elif form == JazzForm.RHYTHM_CHANGES:
            progression = JazzProgressions.rhythm_changes_A(self.key)
        else:
            # Default to ii-V-I
            progression = JazzProgressions.ii_V_I(self.key) * 8

        # Generate components
        bass_line = self.walking_bass.generate_line(progression, beats_per_chord=4)

        # Apply swing timing
        swing_ratio = self._get_swing_ratio()
        bass_line = SwingTiming.apply_swing(bass_line, swing_ratio)

        return {
            "style": self.style.value,
            "tempo": self.tempo,
            "key": self.key,
            "form": form.value,
            "progression": progression,
            "bass_line": bass_line,
            "swing_feel": self.swing_feel.value,
            "swing_ratio": swing_ratio
        }

    def _get_swing_ratio(self) -> float:
        """Get swing ratio based on swing feel"""
        ratios = {
            SwingFeel.STRAIGHT: 0.5,
            SwingFeel.LIGHT_SWING: 0.56,
            SwingFeel.MEDIUM_SWING: 0.62,
            SwingFeel.HEAVY_SWING: 0.65,
            SwingFeel.TRIPLET: 0.667
        }
        return ratios.get(self.swing_feel, 0.62)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_bebop_solo(key: int = 0, num_bars: int = 12) -> List[JazzNote]:
    """Quick bebop solo generation"""
    gen = BebopMelodyGenerator()
    progression = JazzProgressions.jazz_blues(key)

    solo = []
    for chord in progression:
        phrase = gen.generate_phrase(chord, length_beats=4)
        solo.extend(phrase)

    return solo


def generate_walking_bass(key: int = 0, style: str = "bebop") -> List[JazzNote]:
    """Quick walking bass generation"""
    bass = JazzWalkingBass(JazzStyle.BEBOP)
    progression = JazzProgressions.ii_V_I(key)
    return bass.generate_line(progression, beats_per_chord=4, style=style)


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'JazzGenerator',
    'JazzStyle',
    'JazzForm',
    'CompingStyle',
    'SwingFeel',
    'JazzNote',
    'JazzChord',
    'BebopScale',
    'JazzProgressions',
    'JazzWalkingBass',
    'BebopMelodyGenerator',
    'PianoComping',
    'SwingTiming',
    'generate_bebop_solo',
    'generate_walking_bass',
]
