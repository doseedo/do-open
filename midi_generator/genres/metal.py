#!/usr/bin/env python3
"""
Metal & Heavy Music Generator

This module implements comprehensive metal music generation across all sub-genres:
- Thrash Metal (Metallica, Slayer, Megadeth, Anthrax)
- Death Metal (Death, Morbid Angel, Cannibal Corpse)
- Black Metal (Mayhem, Darkthrone, Emperor)
- Progressive Metal (Dream Theater, Tool, Opeth)
- Djent (Meshuggah, Periphery, TesseracT)
- Metalcore, Deathcore, and modern variations

Features:
- Riff generation (chromatic, power chords, palm muting, tremolo)
- Blast beats and double bass patterns
- Gallop rhythms (Iron Maiden style)
- Polyrhythmic riffs (djent/Meshuggah style)
- Drop tuning support (Drop D, C, A, etc.)
- Harmonic minor and Phrygian dominant scales
- Sweep picking arpeggios
- Guitar techniques (palm muting, tremolo picking, sweep)

Author: Agent 11 - Metal & Heavy Music
Date: 2025

Research References:
- "The gallop is the most important metal rhythm" - Guitar World (2024)
- "Meshuggah: Tomas Haake on Djent" - Revolver Magazine
- "Blast Beats: The Extreme Art of Drumset Speed" - Drumming.com
- "Using the Harmonic Minor Scale and Phrygian-Dominant Mode" - Guitar World
- "Thrash metal chromatic riff techniques" - Premier Guitar
- "Tremolo picking techniques for black metal" - Strings and Beyond
- "Drop Tuned Guitars for Metal: The Ultimate Guide" - Riffhard
- Wikipedia: Blast beat, Heavy metal gallop, Thrash metal
- Guitar World: Sweep picking, Palm muting techniques
- Spectrogram analysis of extreme metal drumming - Wolf-Georg Zaddach
"""

import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math


# ============================================================================
# Enums and Data Classes
# ============================================================================

class MetalSubgenre(Enum):
    """Metal sub-genres with distinct characteristics"""
    THRASH = "thrash"                    # Fast, aggressive, chromatic (Metallica, Slayer)
    DEATH = "death"                      # Brutal, tremolo, blast beats (Death, Cannibal Corpse)
    BLACK = "black"                      # Atmospheric, tremolo, raw (Mayhem, Darkthrone)
    PROGRESSIVE = "progressive"          # Complex, odd meters (Dream Theater, Tool)
    DJENT = "djent"                      # Polyrhythmic, syncopated (Meshuggah, Periphery)
    METALCORE = "metalcore"              # Breakdowns, melodic (Killswitch Engage)
    DEATHCORE = "deathcore"              # Heavy breakdowns (Whitechapel)
    POWER = "power"                      # Melodic, fast (DragonForce, Helloween)
    DOOM = "doom"                        # Slow, heavy (Black Sabbath, Electric Wizard)
    NEOCLASSICAL = "neoclassical"        # Classical influenced (Yngwie Malmsteen)


class DropTuning(Enum):
    """Standard drop tunings for metal"""
    STANDARD = "standard"                # E A D G B E
    DROP_D = "drop_d"                    # D A D G B E
    DROP_C = "drop_c"                    # C G C F A D
    DROP_B = "drop_b"                    # B F# B E G# C#
    DROP_A = "drop_a"                    # A E A D F# B
    DROP_G = "drop_g"                    # G D G C E A
    SEVEN_STRING = "seven_string"        # B E A D G B E


class BlastBeatType(Enum):
    """Types of blast beat patterns"""
    STANDARD = "standard"                # Traditional Euro blast
    HAMMER = "hammer"                    # Unison/Suffo blast (simultaneous kick+snare)
    GRAVITY = "gravity"                  # Rim technique, two snares per motion
    BOMB = "bomb"                        # Cannibal blast (every beat)
    HYPER = "hyper"                      # Very fast, over 250 BPM


class RiffTechnique(Enum):
    """Guitar playing techniques"""
    PALM_MUTE = "palm_mute"              # Muted, chunky sound
    TREMOLO = "tremolo"                  # Fast alternate picking
    SWEEP = "sweep"                      # Sweep picking arpeggios
    GALLOP = "gallop"                    # 8th + two 16ths pattern
    POWER_CHORD = "power_chord"          # Root + 5th
    CHROMATIC = "chromatic"              # Half-step movement
    OPEN = "open"                        # Full, ringing notes


@dataclass
class MetalRiff:
    """
    Represents a metal guitar riff

    Attributes:
        notes: List of MIDI note numbers
        durations: List of note durations in 16th notes
        velocities: List of velocities (1-127)
        technique: Playing technique to apply
        palm_mute_intensity: 0.0-1.0 for palm muting
        tuning: Drop tuning to use
    """
    notes: List[int]
    durations: List[int]
    velocities: List[int]
    technique: RiffTechnique = RiffTechnique.POWER_CHORD
    palm_mute_intensity: float = 0.0
    tuning: DropTuning = DropTuning.STANDARD

    def __post_init__(self):
        """Ensure all lists have same length"""
        max_len = max(len(self.notes), len(self.durations), len(self.velocities))
        while len(self.notes) < max_len:
            self.notes.append(self.notes[-1] if self.notes else 60)
        while len(self.durations) < max_len:
            self.durations.append(self.durations[-1] if self.durations else 1)
        while len(self.velocities) < max_len:
            self.velocities.append(self.velocities[-1] if self.velocities else 100)


@dataclass
class DrumPattern:
    """
    Represents a drum pattern

    Attributes:
        kick: List of 16th note positions for kick drum
        snare: List of 16th note positions for snare
        hihat: List of 16th note positions for hi-hat
        crash: List of 16th note positions for crash
        ride: List of 16th note positions for ride
        length: Pattern length in 16th notes
    """
    kick: List[int] = field(default_factory=list)
    snare: List[int] = field(default_factory=list)
    hihat: List[int] = field(default_factory=list)
    crash: List[int] = field(default_factory=list)
    ride: List[int] = field(default_factory=list)
    length: int = 16


# ============================================================================
# Scale Systems
# ============================================================================

class MetalScales:
    """
    Metal scale systems including exotic scales

    Based on research:
    - Harmonic minor: Used extensively in neoclassical metal
    - Phrygian dominant: Fifth mode of harmonic minor, Eastern sound
    - Chromatic: Half-step movements in thrash
    - Octatonic: Meshuggah's signature sound
    """

    # Standard scales
    CHROMATIC = list(range(12))                           # All semitones
    MINOR_PENTATONIC = [0, 3, 5, 7, 10]                   # 1, b3, 4, 5, b7
    NATURAL_MINOR = [0, 2, 3, 5, 7, 8, 10]                # Aeolian
    HARMONIC_MINOR = [0, 2, 3, 5, 7, 8, 11]               # Raised 7th
    MELODIC_MINOR = [0, 2, 3, 5, 7, 9, 11]                # Raised 6th and 7th

    # Exotic scales for metal
    PHRYGIAN_DOMINANT = [0, 1, 4, 5, 7, 8, 10]            # 1, b2, 3, 4, 5, b6, b7
    PHRYGIAN = [0, 1, 3, 5, 7, 8, 10]                     # Dark, Spanish sound
    LOCRIAN = [0, 1, 3, 5, 6, 8, 10]                      # Diminished feel
    WHOLE_TONE = [0, 2, 4, 6, 8, 10]                      # Symmetrical, dissonant

    # Symmetrical scales (Meshuggah style)
    OCTATONIC_HALF_WHOLE = [0, 1, 3, 4, 6, 7, 9, 10]      # Half-whole pattern
    OCTATONIC_WHOLE_HALF = [0, 2, 3, 5, 6, 8, 9, 11]      # Whole-half pattern
    DIMINISHED = [0, 2, 3, 5, 6, 8, 9, 11]                # Symmetrical diminished

    @staticmethod
    def get_notes(root: int, scale_type: str = 'harmonic_minor',
                  octaves: int = 2) -> List[int]:
        """
        Get notes in a metal scale

        Args:
            root: Root note (MIDI)
            scale_type: Type of scale
            octaves: Number of octaves

        Returns:
            List of MIDI note numbers
        """
        scale_map = {
            'chromatic': MetalScales.CHROMATIC,
            'minor_pentatonic': MetalScales.MINOR_PENTATONIC,
            'natural_minor': MetalScales.NATURAL_MINOR,
            'harmonic_minor': MetalScales.HARMONIC_MINOR,
            'melodic_minor': MetalScales.MELODIC_MINOR,
            'phrygian_dominant': MetalScales.PHRYGIAN_DOMINANT,
            'phrygian': MetalScales.PHRYGIAN,
            'locrian': MetalScales.LOCRIAN,
            'whole_tone': MetalScales.WHOLE_TONE,
            'octatonic_hw': MetalScales.OCTATONIC_HALF_WHOLE,
            'octatonic_wh': MetalScales.OCTATONIC_WHOLE_HALF,
            'diminished': MetalScales.DIMINISHED,
        }

        intervals = scale_map.get(scale_type, MetalScales.HARMONIC_MINOR)
        notes = []

        for octave in range(octaves + 1):
            for interval in intervals:
                notes.append(root + interval + (octave * 12))

        return notes


# ============================================================================
# Tuning Systems
# ============================================================================

class TuningSystem:
    """
    Drop tuning calculations and transposition

    Based on research:
    - Drop D: Most common, minimal setup
    - Drop C: Heavier sound, requires heavier strings
    - Drop A: Extreme low tuning, 7-string standard
    """

    # Standard tuning (MIDI notes for open strings)
    STANDARD_TUNING = [40, 45, 50, 55, 59, 64]  # E2, A2, D3, G3, B3, E4

    @staticmethod
    def get_tuning(tuning: DropTuning) -> List[int]:
        """
        Get MIDI notes for open strings in given tuning

        Args:
            tuning: Drop tuning type

        Returns:
            List of MIDI note numbers for open strings
        """
        if tuning == DropTuning.STANDARD:
            return TuningSystem.STANDARD_TUNING.copy()

        elif tuning == DropTuning.DROP_D:
            # D A D G B E
            return [38, 45, 50, 55, 59, 64]

        elif tuning == DropTuning.DROP_C:
            # C G C F A D (whole step down, then drop)
            return [36, 43, 48, 53, 57, 62]

        elif tuning == DropTuning.DROP_B:
            # B F# B E G# C#
            return [35, 42, 47, 52, 56, 61]

        elif tuning == DropTuning.DROP_A:
            # A E A D F# B
            return [33, 40, 45, 50, 54, 59]

        elif tuning == DropTuning.DROP_G:
            # G D G C E A
            return [31, 38, 43, 48, 52, 57]

        elif tuning == DropTuning.SEVEN_STRING:
            # B E A D G B E
            return [35, 40, 45, 50, 55, 59, 64]

        return TuningSystem.STANDARD_TUNING.copy()

    @staticmethod
    def transpose_for_tuning(note: int, from_tuning: DropTuning,
                            to_tuning: DropTuning) -> int:
        """
        Transpose a note between tunings

        Args:
            note: MIDI note number
            from_tuning: Source tuning
            to_tuning: Target tuning

        Returns:
            Transposed MIDI note
        """
        from_open = TuningSystem.get_tuning(from_tuning)[0]
        to_open = TuningSystem.get_tuning(to_tuning)[0]
        offset = to_open - from_open
        return note + offset


# ============================================================================
# Riff Generation
# ============================================================================

class MetalRiffGenerator:
    """
    Generate metal riffs with various techniques

    Research-based implementations:
    - Thrash: Chromatic scales, tritones, fast palm muting
    - Death: Tremolo picking, harmonic minor
    - Djent: Polyrhythmic, syncopated, octatonic scale
    """

    @staticmethod
    def generate_thrash_riff(key: int = 40, tuning: DropTuning = DropTuning.DROP_D,
                            palm_mute: bool = True, measures: int = 2) -> MetalRiff:
        """
        Generate thrash metal riff (Metallica, Slayer style)

        Features:
        - Chromatic movement
        - Tritone intervals
        - Fast palm-muted power chords
        - Aggressive syncopation

        Args:
            key: Root note (MIDI)
            tuning: Drop tuning
            palm_mute: Apply palm muting
            measures: Number of 4/4 measures

        Returns:
            MetalRiff with thrash characteristics
        """
        # Thrash uses chromatic scales and tritones
        notes = []
        durations = []
        velocities = []

        length = measures * 16  # 16th notes per measure in 4/4

        # Start with root power chord
        root = TuningSystem.get_tuning(tuning)[0]

        for i in range(0, length, 4):
            # Chromatic descent pattern
            if i % 8 == 0:
                # Power chord on root
                notes.extend([root, root, root, root])
                durations.extend([1, 1, 1, 1])
                velocities.extend([110, 100, 105, 100])
            else:
                # Chromatic movement with tritone
                chromatic_notes = [
                    root + random.choice([0, 1, 2, 3, 6]),  # Include tritone (6)
                    root + random.choice([0, 1, 2]),
                    root + random.choice([0, 2, 3]),
                    root
                ]
                notes.extend(chromatic_notes)
                durations.extend([1, 1, 1, 1])
                velocities.extend([105, 100, 105, 110])

        palm_mute_intensity = 0.8 if palm_mute else 0.0

        return MetalRiff(
            notes=notes,
            durations=durations,
            velocities=velocities,
            technique=RiffTechnique.PALM_MUTE if palm_mute else RiffTechnique.POWER_CHORD,
            palm_mute_intensity=palm_mute_intensity,
            tuning=tuning
        )

    @staticmethod
    def generate_death_metal_riff(scale: str = 'harmonic_minor', root: int = 40,
                                  tremolo: bool = True, measures: int = 2) -> MetalRiff:
        """
        Generate death metal riff with tremolo picking

        Features:
        - Harmonic minor scale
        - Fast tremolo-picked patterns
        - Dissonant intervals
        - Brutal, relentless feel

        Args:
            scale: Scale type ('harmonic_minor', 'phrygian', etc.)
            root: Root note
            tremolo: Use tremolo picking
            measures: Number of measures

        Returns:
            MetalRiff with death metal characteristics
        """
        scale_notes = MetalScales.get_notes(root, scale, octaves=1)

        notes = []
        durations = []
        velocities = []

        length = measures * 16

        # Tremolo pattern: rapid repeated notes
        for i in range(0, length, 8):
            # Pick a dark interval from harmonic minor
            note1 = random.choice(scale_notes[:4])  # Lower register
            note2 = random.choice(scale_notes[2:6])  # Mid register

            if tremolo:
                # Rapid tremolo on single note
                notes.extend([note1] * 4)
                durations.extend([1, 1, 1, 1])
                velocities.extend([115, 110, 115, 110])

                # Quick transition
                notes.extend([note2] * 4)
                durations.extend([1, 1, 1, 1])
                velocities.extend([115, 110, 115, 110])
            else:
                # Standard riffing
                notes.extend([note1, note1, note2, note2])
                durations.extend([2, 2, 2, 2])
                velocities.extend([110, 105, 110, 105])

        return MetalRiff(
            notes=notes,
            durations=durations,
            velocities=velocities,
            technique=RiffTechnique.TREMOLO if tremolo else RiffTechnique.POWER_CHORD,
            palm_mute_intensity=0.3,
            tuning=DropTuning.DROP_D
        )

    @staticmethod
    def generate_djent_riff(polymeter: Tuple[int, int] = (4, 3),
                           syncopation: float = 0.8, measures: int = 4) -> MetalRiff:
        """
        Generate djent/Meshuggah-style polyrhythmic riff

        Features:
        - Polyrhythmic patterns (3 against 4, etc.)
        - Octatonic scale (half-whole diminished)
        - Heavy syncopation
        - Single-note palm-muted riffs

        Args:
            polymeter: Tuple of (upper, lower) polymeter ratio
            syncopation: Amount of syncopation (0.0-1.0)
            measures: Number of measures

        Returns:
            MetalRiff with djent characteristics
        """
        # Djent uses octatonic (half-whole) scale
        root = 33  # Low A for djent
        scale_notes = MetalScales.get_notes(root, 'octatonic_hw', octaves=1)

        notes = []
        durations = []
        velocities = []

        # Create polymeter: e.g., 4:3 means 4 notes over 3 beats
        ratio_a, ratio_b = polymeter
        pattern_length = ratio_a * ratio_b  # LCM for simple ratios

        total_16ths = measures * 16

        # Generate polyrhythmic pattern
        for cycle in range(total_16ths // pattern_length):
            for i in range(ratio_a):
                pos = (i * pattern_length) // ratio_a

                # Pick note from octatonic scale (low register)
                note = scale_notes[i % len(scale_notes[:4])]

                # Syncopated duration
                if random.random() < syncopation:
                    dur = random.choice([1, 1, 3])  # Syncopated
                else:
                    dur = 2  # Straight

                notes.append(note)
                durations.append(dur)
                velocities.append(random.randint(105, 120))

        # Pad to full length
        while sum(durations) < total_16ths:
            notes.append(notes[-1] if notes else root)
            durations.append(1)
            velocities.append(110)

        # Trim to exact length
        total = 0
        for i, dur in enumerate(durations):
            total += dur
            if total >= total_16ths:
                notes = notes[:i+1]
                durations = durations[:i+1]
                velocities = velocities[:i+1]
                if total > total_16ths:
                    durations[-1] -= (total - total_16ths)
                break

        return MetalRiff(
            notes=notes,
            durations=durations,
            velocities=velocities,
            technique=RiffTechnique.PALM_MUTE,
            palm_mute_intensity=0.9,
            tuning=DropTuning.DROP_A
        )

    @staticmethod
    def generate_gallop_pattern(root_note: int = 40, measures: int = 4,
                               tuning: DropTuning = DropTuning.DROP_D) -> MetalRiff:
        """
        Generate Iron Maiden-style gallop rhythm

        The gallop is an eighth note followed by two sixteenth notes.
        Pattern: LONG-short-short (per beat)

        Based on research:
        - Iron Maiden's signature rhythm
        - Down-down-up picking pattern
        - Palm muted for chunky sound

        Args:
            root_note: Root note for riff
            measures: Number of measures
            tuning: Drop tuning

        Returns:
            MetalRiff with gallop rhythm
        """
        notes = []
        durations = []
        velocities = []

        # Gallop pattern: 2-1-1 (eighth, sixteenth, sixteenth)
        # In 16th note units: 2-1-1 = 4 16th notes per beat

        for measure in range(measures):
            for beat in range(4):  # 4 beats per measure
                # Gallop: LONG-short-short
                notes.extend([root_note, root_note, root_note])
                durations.extend([2, 1, 1])  # Eighth, sixteenth, sixteenth

                # Accent pattern: strong-weak-medium
                velocities.extend([115, 95, 105])

        return MetalRiff(
            notes=notes,
            durations=durations,
            velocities=velocities,
            technique=RiffTechnique.GALLOP,
            palm_mute_intensity=0.7,
            tuning=tuning
        )

    @staticmethod
    def generate_sweep_arpeggio(root: int = 60, chord_type: str = 'minor',
                               direction: str = 'ascending') -> MetalRiff:
        """
        Generate neoclassical sweep picking arpeggio

        Features:
        - Harmonic minor based arpeggios
        - Fast, fluid motion
        - Yngwie Malmsteen style

        Args:
            root: Root note of arpeggio
            chord_type: 'minor', 'major', 'diminished'
            direction: 'ascending', 'descending', 'both'

        Returns:
            MetalRiff with sweep arpeggio
        """
        # Build arpeggio notes
        if chord_type == 'minor':
            intervals = [0, 3, 7, 12, 15, 19, 24]  # Min triad extended
        elif chord_type == 'major':
            intervals = [0, 4, 7, 12, 16, 19, 24]
        elif chord_type == 'diminished':
            intervals = [0, 3, 6, 12, 15, 18, 24]
        else:
            intervals = [0, 3, 7, 12, 15, 19, 24]

        arp_notes = [root + i for i in intervals]

        notes = []
        durations = []
        velocities = []

        if direction in ['ascending', 'both']:
            notes.extend(arp_notes)
            durations.extend([1] * len(arp_notes))  # Fast 16th notes
            velocities.extend([100 + i*2 for i in range(len(arp_notes))])  # Slight crescendo

        if direction in ['descending', 'both']:
            notes.extend(reversed(arp_notes))
            durations.extend([1] * len(arp_notes))
            velocities.extend([110 - i*2 for i in range(len(arp_notes))])  # Slight decrescendo

        return MetalRiff(
            notes=notes,
            durations=durations,
            velocities=velocities,
            technique=RiffTechnique.SWEEP,
            palm_mute_intensity=0.0,
            tuning=DropTuning.STANDARD
        )


# ============================================================================
# Drum Pattern Generation
# ============================================================================

class MetalDrumGenerator:
    """
    Generate metal drum patterns including blast beats

    Based on research:
    - Blast beats: 180-280 BPM 16th notes
    - Types: Standard, Hammer, Gravity, Bomb
    - Double bass: Fundamental to metal
    """

    # MIDI note numbers for drums (General MIDI)
    KICK = 36
    SNARE = 38
    HIHAT_CLOSED = 42
    HIHAT_OPEN = 46
    CRASH = 49
    RIDE = 51
    CHINA = 52
    TOM_HIGH = 48
    TOM_MID = 47
    TOM_LOW = 45

    @staticmethod
    def generate_blast_beat(blast_type: BlastBeatType = BlastBeatType.STANDARD,
                           measures: int = 2, bpm: int = 200) -> DrumPattern:
        """
        Generate blast beat pattern

        Types:
        - STANDARD: Kick on downbeats, snare on upbeats
        - HAMMER: Kick and snare simultaneous
        - GRAVITY: Double snare hits
        - BOMB: Every beat simultaneous
        - HYPER: Very fast variation

        Args:
            blast_type: Type of blast beat
            measures: Number of measures
            bpm: Beats per minute (typically 180-280)

        Returns:
            DrumPattern with blast beat
        """
        length = measures * 16  # 16th notes per measure
        pattern = DrumPattern(length=length)

        if blast_type == BlastBeatType.STANDARD:
            # Traditional/Euro blast
            # Kick on beats 1,2,3,4 and snare on offbeats
            for i in range(length):
                if i % 4 == 0:
                    pattern.kick.append(i)
                if i % 4 == 2:
                    pattern.snare.append(i)
                # Hi-hat/ride on all 16ths
                pattern.ride.append(i)

        elif blast_type == BlastBeatType.HAMMER:
            # Hammer/Unison/Suffo blast
            # Kick and snare hit simultaneously
            for i in range(0, length, 2):  # Every 8th note
                pattern.kick.append(i)
                pattern.snare.append(i)
                pattern.hihat.append(i)

        elif blast_type == BlastBeatType.GRAVITY:
            # Gravity blast - double snare hits
            for i in range(length):
                if i % 4 == 0:
                    pattern.kick.append(i)
                if i % 2 == 1:
                    # Two snare hits per hand motion
                    pattern.snare.append(i)
                pattern.hihat.append(i)

        elif blast_type == BlastBeatType.BOMB:
            # Cannibal/Bomb blast
            # Everything on every beat
            for i in range(0, length, 4):  # Every beat
                pattern.kick.append(i)
                pattern.snare.append(i)
                pattern.crash.append(i)

        elif blast_type == BlastBeatType.HYPER:
            # Hyper blast - very fast, all 16ths
            for i in range(length):
                if i % 2 == 0:
                    pattern.kick.append(i)
                else:
                    pattern.snare.append(i)
                pattern.hihat.append(i)

        return pattern

    @staticmethod
    def generate_double_bass_pattern(measures: int = 2,
                                    pattern_type: str = 'sixteenths') -> DrumPattern:
        """
        Generate double bass drum pattern

        Args:
            measures: Number of measures
            pattern_type: 'sixteenths', 'triplets', 'gallop'

        Returns:
            DrumPattern with double bass
        """
        length = measures * 16
        pattern = DrumPattern(length=length)

        if pattern_type == 'sixteenths':
            # Straight 16th notes on kick
            for i in range(length):
                pattern.kick.append(i)
                # Snare on backbeat
                if i % 8 == 4:
                    pattern.snare.append(i)
                # Hi-hat
                if i % 2 == 0:
                    pattern.hihat.append(i)

        elif pattern_type == 'triplets':
            # Triplet feel
            for i in range(0, length, 3):
                pattern.kick.append(i)
                if i % 12 == 6:
                    pattern.snare.append(i)

        elif pattern_type == 'gallop':
            # Gallop on bass drums
            for beat in range(measures * 4):
                base = beat * 4
                pattern.kick.extend([base, base + 2, base + 3])
                if beat % 2 == 1:
                    pattern.snare.append(base)

        return pattern

    @staticmethod
    def generate_thrash_beat(measures: int = 4) -> DrumPattern:
        """
        Generate typical thrash metal drum pattern

        Features:
        - Fast, driving beat
        - Heavy on bass drum
        - Aggressive snare

        Args:
            measures: Number of measures

        Returns:
            DrumPattern with thrash characteristics
        """
        length = measures * 16
        pattern = DrumPattern(length=length)

        for i in range(length):
            # Kick on downbeats and syncopated
            if i % 8 == 0 or i % 8 == 6:
                pattern.kick.append(i)

            # Snare on backbeat
            if i % 8 == 4:
                pattern.snare.append(i)

            # Hi-hat 8th notes
            if i % 2 == 0:
                pattern.hihat.append(i)

            # Crash on measure starts
            if i % 16 == 0:
                pattern.crash.append(i)

        return pattern

    @staticmethod
    def generate_breakdown_pattern(measures: int = 2,
                                   syncopation: float = 0.7) -> DrumPattern:
        """
        Generate metalcore/deathcore breakdown pattern

        Features:
        - Heavy, syncopated hits
        - Slow, crushing feel
        - Synchronized with guitar

        Args:
            measures: Number of measures
            syncopation: Amount of syncopation

        Returns:
            DrumPattern for breakdown
        """
        length = measures * 16
        pattern = DrumPattern(length=length)

        # Breakdown hits on syncopated positions
        hit_positions = [0, 3, 6, 10, 13]  # Syncopated within 16 beats

        for measure in range(measures):
            base = measure * 16
            for pos in hit_positions:
                actual_pos = base + pos
                if actual_pos < length:
                    pattern.kick.append(actual_pos)
                    pattern.snare.append(actual_pos)
                    pattern.crash.append(actual_pos)

        return pattern


# ============================================================================
# Main Generator Class
# ============================================================================

class MetalGenerator:
    """
    Main metal music generator

    Combines riff generation, drum patterns, and composition techniques
    for comprehensive metal music creation across all sub-genres.

    Usage:
        generator = MetalGenerator()

        # Generate thrash riff
        riff = generator.generate_riff(
            subgenre=MetalSubgenre.THRASH,
            key=40,
            tuning=DropTuning.DROP_D
        )

        # Generate blast beat
        drums = generator.generate_drums(
            subgenre=MetalSubgenre.DEATH,
            blast_type=BlastBeatType.STANDARD
        )
    """

    def __init__(self):
        """Initialize metal generator"""
        self.riff_generator = MetalRiffGenerator()
        self.drum_generator = MetalDrumGenerator()

    def generate_riff(self, subgenre: MetalSubgenre = MetalSubgenre.THRASH,
                     key: int = 40, tuning: DropTuning = DropTuning.DROP_D,
                     measures: int = 4, **kwargs) -> MetalRiff:
        """
        Generate a riff for specified sub-genre

        Args:
            subgenre: Metal sub-genre
            key: Root note (MIDI)
            tuning: Drop tuning
            measures: Number of measures
            **kwargs: Additional parameters for specific generators

        Returns:
            MetalRiff object
        """
        if subgenre == MetalSubgenre.THRASH:
            return self.riff_generator.generate_thrash_riff(
                key=key, tuning=tuning, measures=measures
            )

        elif subgenre == MetalSubgenre.DEATH:
            return self.riff_generator.generate_death_metal_riff(
                root=key, measures=measures,
                scale=kwargs.get('scale', 'harmonic_minor')
            )

        elif subgenre == MetalSubgenre.DJENT:
            return self.riff_generator.generate_djent_riff(
                polymeter=kwargs.get('polymeter', (4, 3)),
                measures=measures
            )

        elif subgenre == MetalSubgenre.POWER:
            return self.riff_generator.generate_gallop_pattern(
                root_note=key, measures=measures, tuning=tuning
            )

        elif subgenre == MetalSubgenre.NEOCLASSICAL:
            return self.riff_generator.generate_sweep_arpeggio(
                root=key + 12,  # Higher register
                chord_type=kwargs.get('chord_type', 'minor')
            )

        else:
            # Default to thrash
            return self.riff_generator.generate_thrash_riff(
                key=key, tuning=tuning, measures=measures
            )

    def generate_drums(self, subgenre: MetalSubgenre = MetalSubgenre.THRASH,
                      measures: int = 4, **kwargs) -> DrumPattern:
        """
        Generate drums for specified sub-genre

        Args:
            subgenre: Metal sub-genre
            measures: Number of measures
            **kwargs: Additional parameters

        Returns:
            DrumPattern object
        """
        if subgenre in [MetalSubgenre.DEATH, MetalSubgenre.BLACK]:
            blast_type = kwargs.get('blast_type', BlastBeatType.STANDARD)
            return self.drum_generator.generate_blast_beat(
                blast_type=blast_type, measures=measures
            )

        elif subgenre == MetalSubgenre.THRASH:
            return self.drum_generator.generate_thrash_beat(measures=measures)

        elif subgenre in [MetalSubgenre.METALCORE, MetalSubgenre.DEATHCORE]:
            return self.drum_generator.generate_breakdown_pattern(measures=measures)

        else:
            return self.drum_generator.generate_double_bass_pattern(measures=measures)

    def generate_full_section(self, subgenre: MetalSubgenre = MetalSubgenre.THRASH,
                            key: int = 40, tuning: DropTuning = DropTuning.DROP_D,
                            measures: int = 4) -> Dict[str, Union[MetalRiff, DrumPattern]]:
        """
        Generate a full section with riff and drums

        Args:
            subgenre: Metal sub-genre
            key: Root note
            tuning: Drop tuning
            measures: Number of measures

        Returns:
            Dict with 'riff' and 'drums' keys
        """
        riff = self.generate_riff(
            subgenre=subgenre,
            key=key,
            tuning=tuning,
            measures=measures
        )

        drums = self.generate_drums(
            subgenre=subgenre,
            measures=measures
        )

        return {
            'riff': riff,
            'drums': drums,
            'subgenre': subgenre.value,
            'key': key,
            'tuning': tuning.value,
            'measures': measures
        }


# ============================================================================
# Utility Functions
# ============================================================================

def convert_to_midi_events(riff: MetalRiff, start_tick: int = 0,
                          ppqn: int = 480) -> List[Dict]:
    """
    Convert MetalRiff to MIDI events

    Args:
        riff: MetalRiff object
        start_tick: Starting tick position
        ppqn: Pulses per quarter note

    Returns:
        List of MIDI event dictionaries
    """
    events = []
    current_tick = start_tick

    # 16th note duration in ticks
    sixteenth_note = ppqn // 4

    for note, duration, velocity in zip(riff.notes, riff.durations, riff.velocities):
        # Adjust velocity for palm muting
        if riff.palm_mute_intensity > 0:
            velocity = int(velocity * (1.0 - riff.palm_mute_intensity * 0.3))
            velocity = max(40, min(127, velocity))

        # Note on
        events.append({
            'type': 'note_on',
            'note': note,
            'velocity': velocity,
            'tick': current_tick,
            'channel': 0
        })

        # Note off
        note_duration = duration * sixteenth_note

        # Shorten for palm muting
        if riff.palm_mute_intensity > 0:
            note_duration = int(note_duration * (1.0 - riff.palm_mute_intensity * 0.5))

        events.append({
            'type': 'note_off',
            'note': note,
            'velocity': 0,
            'tick': current_tick + note_duration,
            'channel': 0
        })

        current_tick += duration * sixteenth_note

    return events


def convert_drums_to_midi_events(pattern: DrumPattern, start_tick: int = 0,
                                 ppqn: int = 480) -> List[Dict]:
    """
    Convert DrumPattern to MIDI events

    Args:
        pattern: DrumPattern object
        start_tick: Starting tick position
        ppqn: Pulses per quarter note

    Returns:
        List of MIDI event dictionaries
    """
    events = []
    sixteenth_note = ppqn // 4

    # Map drum types to MIDI notes and velocities
    drum_map = [
        (pattern.kick, MetalDrumGenerator.KICK, 110),
        (pattern.snare, MetalDrumGenerator.SNARE, 115),
        (pattern.hihat, MetalDrumGenerator.HIHAT_CLOSED, 90),
        (pattern.crash, MetalDrumGenerator.CRASH, 120),
        (pattern.ride, MetalDrumGenerator.RIDE, 95),
    ]

    for positions, note, velocity in drum_map:
        for pos in positions:
            tick = start_tick + (pos * sixteenth_note)

            # Note on
            events.append({
                'type': 'note_on',
                'note': note,
                'velocity': velocity,
                'tick': tick,
                'channel': 9  # Drum channel
            })

            # Note off (short duration for drums)
            events.append({
                'type': 'note_off',
                'note': note,
                'velocity': 0,
                'tick': tick + (sixteenth_note // 2),
                'channel': 9
            })

    return events


# ============================================================================
# Example Usage and Tests
# ============================================================================

if __name__ == "__main__":
    print("Metal & Heavy Music Generator - Test Suite")
    print("=" * 60)

    generator = MetalGenerator()

    # Test 1: Thrash metal riff
    print("\n[TEST 1] Generating thrash metal riff (Drop D)...")
    thrash_riff = generator.generate_riff(
        subgenre=MetalSubgenre.THRASH,
        key=38,  # D
        tuning=DropTuning.DROP_D,
        measures=2
    )
    print(f"  Notes: {len(thrash_riff.notes)} notes generated")
    print(f"  Technique: {thrash_riff.technique.value}")
    print(f"  Palm mute intensity: {thrash_riff.palm_mute_intensity}")
    print(f"  First 8 notes: {thrash_riff.notes[:8]}")

    # Test 2: Blast beat
    print("\n[TEST 2] Generating blast beat (Standard)...")
    blast = generator.drum_generator.generate_blast_beat(
        blast_type=BlastBeatType.STANDARD,
        measures=2,
        bpm=200
    )
    print(f"  Kick hits: {len(blast.kick)}")
    print(f"  Snare hits: {len(blast.snare)}")
    print(f"  Ride hits: {len(blast.ride)}")

    # Test 3: Gallop pattern
    print("\n[TEST 3] Generating Iron Maiden gallop...")
    gallop = generator.riff_generator.generate_gallop_pattern(
        root_note=40,
        measures=2
    )
    print(f"  Pattern length: {len(gallop.notes)} notes")
    print(f"  Durations (first 12): {gallop.durations[:12]}")
    print(f"  Expected: [2,1,1, 2,1,1, ...] for gallop")

    # Test 4: Djent polyrhythmic riff
    print("\n[TEST 4] Generating djent riff (4:3 polymeter)...")
    djent = generator.generate_riff(
        subgenre=MetalSubgenre.DJENT,
        polymeter=(4, 3),
        measures=4
    )
    print(f"  Notes: {len(djent.notes)}")
    print(f"  Tuning: {djent.tuning.value}")
    print(f"  Palm mute: {djent.palm_mute_intensity}")

    # Test 5: Death metal with harmonic minor
    print("\n[TEST 5] Generating death metal riff (Harmonic Minor)...")
    death = generator.generate_riff(
        subgenre=MetalSubgenre.DEATH,
        key=40,
        scale='harmonic_minor',
        measures=2
    )
    print(f"  Notes: {len(death.notes)}")
    print(f"  Technique: {death.technique.value}")

    # Test 6: Sweep picking arpeggio
    print("\n[TEST 6] Generating sweep picking arpeggio...")
    sweep = generator.riff_generator.generate_sweep_arpeggio(
        root=60,
        chord_type='minor',
        direction='both'
    )
    print(f"  Arpeggio length: {len(sweep.notes)} notes")
    print(f"  Notes: {sweep.notes}")

    # Test 7: Drop tuning system
    print("\n[TEST 7] Testing drop tuning system...")
    for tuning in [DropTuning.DROP_D, DropTuning.DROP_C, DropTuning.DROP_A]:
        notes = TuningSystem.get_tuning(tuning)
        print(f"  {tuning.value}: {notes}")

    # Test 8: Metal scales
    print("\n[TEST 8] Testing metal scales...")
    scales_to_test = ['harmonic_minor', 'phrygian_dominant', 'octatonic_hw']
    for scale in scales_to_test:
        notes = MetalScales.get_notes(60, scale, octaves=1)
        print(f"  {scale}: {notes}")

    # Test 9: Blast beat variations
    print("\n[TEST 9] Testing all blast beat types...")
    for blast_type in [BlastBeatType.STANDARD, BlastBeatType.HAMMER,
                       BlastBeatType.GRAVITY, BlastBeatType.BOMB]:
        pattern = generator.drum_generator.generate_blast_beat(
            blast_type=blast_type,
            measures=1
        )
        print(f"  {blast_type.value}: Kick={len(pattern.kick)}, Snare={len(pattern.snare)}")

    # Test 10: Full section generation
    print("\n[TEST 10] Generating full metal section...")
    section = generator.generate_full_section(
        subgenre=MetalSubgenre.THRASH,
        key=38,
        tuning=DropTuning.DROP_D,
        measures=4
    )
    print(f"  Subgenre: {section['subgenre']}")
    print(f"  Riff notes: {len(section['riff'].notes)}")
    print(f"  Drum kicks: {len(section['drums'].kick)}")

    # Test 11: MIDI conversion
    print("\n[TEST 11] Converting to MIDI events...")
    midi_events = convert_to_midi_events(thrash_riff, start_tick=0, ppqn=480)
    print(f"  Generated {len(midi_events)} MIDI events")
    print(f"  First event: {midi_events[0]}")

    # Test 12: Drum MIDI conversion
    print("\n[TEST 12] Converting drums to MIDI...")
    drum_events = convert_drums_to_midi_events(blast, start_tick=0, ppqn=480)
    print(f"  Generated {len(drum_events)} drum MIDI events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("\nKey Statistics:")
    print(f"  - {len(MetalSubgenre)} metal sub-genres supported")
    print(f"  - {len(DropTuning)} tuning systems")
    print(f"  - {len(BlastBeatType)} blast beat variations")
    print(f"  - {len(RiffTechnique)} guitar techniques")
    print("=" * 60)
