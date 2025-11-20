#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Instrumentation Specialist Engine - Agent 21
=============================================

Intelligent instrument selection, orchestration, and voicing for multi-genre music generation.

This module provides comprehensive instrumentation expertise across musical styles,
combining classical orchestration principles, jazz arranging techniques, and modern
production practices.

Research Foundation:
-------------------
- Orchestration: Rimsky-Korsakov, Samuel Adler, Walter Piston
- Jazz Arranging: Sammy Nestico, Maria Schneider, Bob Brookmeyer
- Big Band: Duke Ellington, Count Basie, Thad Jones/Mel Lewis
- Piano Voicings: Bill Evans, McCoy Tyner, Herbie Hancock
- Bass Patterns: Ray Brown, Ron Carter, Paul Chambers
- Drum Patterns: Max Roach, Tony Williams, Elvin Jones

Features:
---------
1. **Instrument Selection**
   - Genre-appropriate ensembles
   - Texture density control
   - Role-based distribution (melody, harmony, bass, rhythm)

2. **Orchestration**
   - Range-aware voice distribution
   - Blend compatibility analysis
   - Doubling recommendations
   - Dynamic balance

3. **Voicing Patterns**
   - Piano: Drop-2, drop-3, rootless, quartal, clusters
   - Brass: Section voicings, spread vs. closed
   - Strings: Divisi, unison, harmonics
   - Woodwinds: Choir balance, blend optimization

4. **Bass Patterns**
   - Walking bass lines (jazz)
   - Pedal tones
   - Contrary motion
   - Scalar patterns

5. **Drum Patterns**
   - Genre-specific grooves (swing, rock, funk, Latin)
   - Fill patterns
   - Dynamics and articulation

6. **Articulation Assignment**
   - Instrument-family-specific articulations
   - Genre conventions
   - Musical context awareness

Author: Agent 21 - Instrumentation Specialist
Date: 2025
License: MIT
"""

import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from enum import Enum
import math

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from core
from core.instrument_library import (
    Instrument, InstrumentFamily, InstrumentRange, ArticulationType,
    get_instrument, get_instruments_by_family, is_in_comfortable_range,
    is_in_optimal_range, INSTRUMENTS
)

# ============================================================================
# ENUMERATIONS
# ============================================================================

class InstrumentRole(Enum):
    """Functional roles instruments can play in an ensemble"""
    MELODY = "melody"
    HARMONY = "harmony"
    BASS = "bass"
    RHYTHM = "rhythm"
    COLOR = "color"
    COUNTER_MELODY = "counter_melody"
    PEDAL = "pedal"
    OBBLIGATO = "obbligato"


class EnsembleType(Enum):
    """Types of musical ensembles"""
    SOLO = "solo"
    DUO = "duo"
    TRIO = "trio"
    QUARTET = "quartet"
    QUINTET = "quintet"
    SEXTET = "sextet"
    JAZZ_COMBO = "jazz_combo"
    BIG_BAND = "big_band"
    CHAMBER_ORCHESTRA = "chamber_orchestra"
    SYMPHONY_ORCHESTRA = "symphony_orchestra"
    STRING_QUARTET = "string_quartet"
    BRASS_QUINTET = "brass_quintet"
    WOODWIND_QUINTET = "woodwind_quintet"
    ROCK_BAND = "rock_band"
    POP_BAND = "pop_band"


class VoicingType(Enum):
    """Types of harmonic voicings"""
    # Piano/Keyboard
    ROOT_POSITION = "root_position"
    FIRST_INVERSION = "first_inversion"
    SECOND_INVERSION = "second_inversion"
    DROP_2 = "drop_2"
    DROP_3 = "drop_3"
    DROP_2_4 = "drop_2_4"
    ROOTLESS = "rootless"
    QUARTAL = "quartal"
    CLUSTER = "cluster"
    SHELL = "shell"

    # Horn sections
    CLOSE_POSITION = "close_position"
    SPREAD_POSITION = "spread_position"
    FOUR_WAY_CLOSE = "four_way_close"
    FOUR_WAY_OPEN = "four_way_open"
    FIVE_WAY_CLOSE = "five_way_close"
    DOUBLE_LEAD = "double_lead"

    # Strings
    DIVISI = "divisi"
    UNISON = "unison"
    OCTAVES = "octaves"


class TextureDensity(Enum):
    """Texture density levels"""
    SPARSE = "sparse"          # Minimal instrumentation
    LIGHT = "light"            # Light texture
    MEDIUM = "medium"          # Moderate density
    FULL = "full"              # Full ensemble
    DENSE = "dense"            # Very dense orchestration


class BassPattern(Enum):
    """Bass line pattern types"""
    WALKING = "walking"                # Jazz walking bass
    PEDAL = "pedal"                    # Sustained pedal tone
    OSTINATO = "ostinato"              # Repeated pattern
    SCALAR = "scalar"                  # Scalar motion
    ARPEGGIO = "arpeggio"              # Arpeggiated
    CONTRARY_MOTION = "contrary"       # Contrary to melody
    ROOTS = "roots"                    # Root notes only
    CHROMATIC = "chromatic"            # Chromatic approach
    TWO_FEEL = "two_feel"              # Two beats per bar
    LATIN = "latin"                    # Latin patterns
    FUNK = "funk"                      # Funk syncopation


class DrumPattern(Enum):
    """Drum pattern types"""
    # Jazz
    SWING = "swing"
    BEBOP = "bebop"
    LATIN_JAZZ = "latin_jazz"
    BOSSA_NOVA = "bossa_nova"
    SAMBA = "samba"
    AFRO_CUBAN = "afro_cuban"

    # Rock/Pop
    ROCK_BASIC = "rock_basic"
    ROCK_SHUFFLE = "rock_shuffle"
    POP_BASIC = "pop_basic"
    BALLAD = "ballad"

    # Funk/R&B
    FUNK = "funk"
    SECOND_LINE = "second_line"
    HIP_HOP = "hip_hop"

    # Other
    WALTZ = "waltz"
    MARCH = "march"
    BRUSHES = "brushes"


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class InstrumentationProfile:
    """
    Complete instrumentation specification for a piece or section
    """
    ensemble_type: EnsembleType
    instruments: List[Instrument]
    roles: Dict[str, InstrumentRole]  # instrument_name -> role
    voicing_type: VoicingType = VoicingType.DROP_2
    texture_density: TextureDensity = TextureDensity.MEDIUM
    bass_pattern: BassPattern = BassPattern.WALKING
    drum_pattern: Optional[DrumPattern] = None

    # Orchestration details
    doublings: Dict[str, List[str]] = field(default_factory=dict)  # part -> [instruments]
    ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # instrument -> (low, high)
    dynamics: Dict[str, int] = field(default_factory=dict)  # instrument -> velocity
    articulations: Dict[str, ArticulationType] = field(default_factory=dict)

    # Blend and balance
    blend_score: float = 0.0  # 0.0 = poor blend, 1.0 = perfect blend
    balance_ratios: Dict[str, float] = field(default_factory=dict)  # instrument -> relative volume


@dataclass
class VoicingSpec:
    """
    Specification for a harmonic voicing
    """
    voicing_type: VoicingType
    notes: List[int]  # MIDI note numbers from low to high
    root: int  # Root note
    chord_tones: List[int]  # Which scale degrees (0-based from root)
    extensions: List[int] = field(default_factory=list)  # Extended chord tones (7, 9, 11, 13)
    omissions: List[int] = field(default_factory=list)  # Omitted chord tones
    doublings: List[int] = field(default_factory=list)  # Doubled notes
    inversion: int = 0  # 0 = root position, 1 = first inversion, etc.
    spread: int = 0  # Octave spread (0 = close, 1+ = spread by octaves)


@dataclass
class OrchestrationRule:
    """
    A rule for orchestration decisions
    """
    name: str
    condition: str  # Description of when this rule applies
    recommendation: str  # What to do
    priority: int = 1  # Higher = more important
    genre_specific: List[str] = field(default_factory=list)


@dataclass
class BlendCompatibility:
    """
    Blend compatibility between two instruments
    """
    instrument_a: str
    instrument_b: str
    compatibility_score: float  # 0.0 = poor, 1.0 = excellent
    register: str  # "low", "mid", "high"
    context: str  # When this compatibility applies
    notes: str = ""  # Additional notes


@dataclass
class EnsembleTemplate:
    """
    Template for a standard ensemble configuration
    """
    name: str
    ensemble_type: EnsembleType
    instruments: List[str]  # Instrument names
    typical_roles: Dict[str, InstrumentRole]
    voicing_preferences: List[VoicingType]
    genre_associations: List[str]
    texture_range: Tuple[TextureDensity, TextureDensity]  # (min, max)


# ============================================================================
# ORCHESTRATION KNOWLEDGE BASE
# ============================================================================

class OrchestrationKnowledge:
    """
    Comprehensive orchestration knowledge database
    """

    # Classical orchestration principles
    BLEND_COMPATIBILITY_DB = [
        # Excellent blends
        BlendCompatibility("Flute", "Clarinet", 0.95, "mid", "Unison or octaves",
                          "Classic woodwind blend"),
        BlendCompatibility("Oboe", "English Horn", 0.98, "mid", "Any register",
                          "Same family, natural blend"),
        BlendCompatibility("Violin", "Viola", 0.99, "mid", "String section",
                          "Perfect string blend"),
        BlendCompatibility("Trumpet", "Trombone", 0.90, "mid", "Brass section",
                          "Classic brass section sound"),
        BlendCompatibility("Flute", "Violin", 0.85, "high", "Upper register",
                          "Ethereal combination"),

        # Good blends
        BlendCompatibility("Clarinet", "Bassoon", 0.80, "low", "Low register",
                          "Dark woodwind color"),
        BlendCompatibility("Horn", "Cello", 0.88, "mid", "Melodic lines",
                          "Rich, warm blend"),
        BlendCompatibility("Trumpet", "Alto Saxophone", 0.75, "mid", "Jazz sections",
                          "Big band blend"),
        BlendCompatibility("Flute", "Harp", 0.82, "high", "Delicate passages",
                          "Impressionistic color"),

        # Contrasting (intentional)
        BlendCompatibility("Piccolo", "Tuba", 0.30, "all", "Extreme registers",
                          "Maximum contrast"),
        BlendCompatibility("Trumpet", "Flute", 0.40, "mid", "Different attacks",
                          "Use for contrast, not blend"),
    ]

    # Orchestration rules from treatises
    ORCHESTRATION_RULES = [
        OrchestrationRule(
            "Rimsky-Korsakov: Brass Doubling",
            "When brass doubles strings",
            "Place brass below strings to avoid covering them",
            priority=9
        ),
        OrchestrationRule(
            "Adler: Woodwind Balance",
            "Oboe in woodwind chord",
            "Oboe has penetrating tone - use sparingly in middle voices",
            priority=7
        ),
        OrchestrationRule(
            "Berlioz: Extreme Registers",
            "Very high or very low notes",
            "Use sparingly; they fatigue performers and lose clarity",
            priority=8
        ),
        OrchestrationRule(
            "Piston: String Divisi",
            "String section divides into parts",
            "More divisi = thinner sound. Limit to 4-way max in smaller sections",
            priority=6
        ),
        OrchestrationRule(
            "Jazz: Lead Trumpet",
            "Trumpet leads brass section",
            "Lead trumpet typically plays highest note in close voicings",
            priority=8,
            genre_specific=["jazz", "big_band"]
        ),
        OrchestrationRule(
            "Jazz: Saxophone Soli",
            "Saxophone section alone",
            "Use close voicings (within an octave) for classic sax soli sound",
            priority=7,
            genre_specific=["jazz", "big_band"]
        ),
        OrchestrationRule(
            "Pop: Guitar Voicings",
            "Guitar in pop/rock context",
            "Avoid close voicings below middle C - they sound muddy",
            priority=7,
            genre_specific=["pop", "rock"]
        ),
    ]

    # Doubling guidelines
    DOUBLING_RULES = {
        "unison": {
            "description": "Same pitch, different instruments",
            "good_for": ["Reinforcement", "Timbral blend", "Emphasis"],
            "avoid": ["Oboe + Clarinet in low register (tone clash)"],
        },
        "octaves": {
            "description": "One or more octaves apart",
            "good_for": ["Strength", "Fullness", "Registral expansion"],
            "avoid": ["Too many octaves (sounds hollow)"],
        },
        "doubling_at_third": {
            "description": "Parallel thirds or sixths",
            "good_for": ["Warmth", "Thickness", "Melodic lines"],
            "classic": ["Flute + Clarinet", "Violin I + Violin II"],
        },
    }


# ============================================================================
# PIANO VOICING GENERATOR
# ============================================================================

class PianoVoicingGenerator:
    """
    Generates idiomatic piano voicings across styles

    Implements voicing techniques from:
    - Bill Evans (rootless voicings)
    - McCoy Tyner (quartal voicings)
    - Oscar Peterson (block chords)
    - Herbie Hancock (clusters, modern harmony)
    """

    @staticmethod
    def generate_drop_2(root: int, chord_tones: List[int], extensions: List[int] = None) -> VoicingSpec:
        """
        Generate drop-2 voicing (2nd note from top dropped an octave)

        Standard jazz piano/guitar voicing. Creates good voice leading
        and avoids muddy low intervals.

        Args:
            root: Root note MIDI number
            chord_tones: List of intervals from root [0, 4, 7] for major triad
            extensions: Optional extensions like [10] for 7th, [14] for 9th

        Returns:
            VoicingSpec with drop-2 voicing
        """
        if extensions is None:
            extensions = []

        # Build close position chord
        all_tones = sorted(set(chord_tones + extensions))
        close_voicing = [root + tone for tone in all_tones]

        # Ensure we have at least 4 notes for drop-2
        while len(close_voicing) < 4:
            close_voicing.append(close_voicing[-1] + 12)  # Add octave

        # Apply drop-2: take second from top, drop an octave
        drop_2_voicing = close_voicing[:]
        if len(drop_2_voicing) >= 4:
            second_from_top = drop_2_voicing[-2]
            drop_2_voicing.remove(second_from_top)
            drop_2_voicing.insert(0, second_from_top - 12)

        drop_2_voicing.sort()

        return VoicingSpec(
            voicing_type=VoicingType.DROP_2,
            notes=drop_2_voicing,
            root=root,
            chord_tones=chord_tones,
            extensions=extensions,
            doublings=[],
            inversion=0,
            spread=1
        )

    @staticmethod
    def generate_drop_3(root: int, chord_tones: List[int], extensions: List[int] = None) -> VoicingSpec:
        """
        Generate drop-3 voicing (3rd note from top dropped an octave)

        Creates wider spread, often used for guitar and piano.
        """
        if extensions is None:
            extensions = []

        all_tones = sorted(set(chord_tones + extensions))
        close_voicing = [root + tone for tone in all_tones]

        while len(close_voicing) < 4:
            close_voicing.append(close_voicing[-1] + 12)

        drop_3_voicing = close_voicing[:]
        if len(drop_3_voicing) >= 4:
            third_from_top = drop_3_voicing[-3]
            drop_3_voicing.remove(third_from_top)
            drop_3_voicing.insert(0, third_from_top - 12)

        drop_3_voicing.sort()

        return VoicingSpec(
            voicing_type=VoicingType.DROP_3,
            notes=drop_3_voicing,
            root=root,
            chord_tones=chord_tones,
            extensions=extensions,
            doublings=[],
            inversion=0,
            spread=1
        )

    @staticmethod
    def generate_rootless(root: int, chord_tones: List[int], extensions: List[int],
                         type_a: bool = True) -> VoicingSpec:
        """
        Generate rootless voicing (Bill Evans style)

        Omits root (bass plays it), uses 3rd, 7th, and extensions.
        Two types:
        - Type A: 3rd in bass (for root position and 2nd inversion chords)
        - Type B: 7th in bass (for 1st inversion chords)

        Args:
            root: Root note
            chord_tones: Chord tones (will omit root)
            extensions: Extensions (9, 11, 13)
            type_a: True for Type A, False for Type B
        """
        # Filter out root (0) from chord tones
        tones_no_root = [t for t in chord_tones if t % 12 != 0]
        all_tones = sorted(set(tones_no_root + extensions))

        # Build voicing
        notes = [root + tone for tone in all_tones]

        # Type A: 3rd on bottom
        # Type B: 7th on bottom
        if not type_a and len(notes) >= 2:
            # Move 7th to bottom (assuming 7th is present)
            seventh_candidates = [n for n in notes if (n - root) % 12 in [10, 11]]  # m7 or M7
            if seventh_candidates:
                seventh = seventh_candidates[0]
                notes.remove(seventh)
                notes.insert(0, seventh - 12 if seventh - 12 >= root - 12 else seventh)

        notes.sort()

        return VoicingSpec(
            voicing_type=VoicingType.ROOTLESS,
            notes=notes,
            root=root,
            chord_tones=tones_no_root,
            extensions=extensions,
            omissions=[0],  # Root omitted
            doublings=[],
            inversion=0,
            spread=0
        )

    @staticmethod
    def generate_quartal(root: int, num_voices: int = 4) -> VoicingSpec:
        """
        Generate quartal voicing (McCoy Tyner style)

        Built from perfect 4ths instead of thirds. Creates modern, open sound.
        Common in modal jazz and contemporary classical.

        Args:
            root: Root note
            num_voices: Number of voices (typically 3-5)
        """
        notes = [root]
        current = root

        # Stack perfect 4ths
        for _ in range(num_voices - 1):
            current += 5  # Perfect 4th
            notes.append(current)

        return VoicingSpec(
            voicing_type=VoicingType.QUARTAL,
            notes=notes,
            root=root,
            chord_tones=[0, 5, 10],  # Approximate chord tones
            extensions=[],
            doublings=[],
            inversion=0,
            spread=0
        )

    @staticmethod
    def generate_cluster(root: int, num_voices: int = 4, whole_tone: bool = False) -> VoicingSpec:
        """
        Generate cluster voicing (Herbie Hancock style)

        Tight chromatic or whole-tone clusters. Creates modern, dense harmony.

        Args:
            root: Root note
            num_voices: Number of voices
            whole_tone: True for whole-tone scale, False for chromatic
        """
        notes = [root]
        current = root
        interval = 2 if whole_tone else 1

        for _ in range(num_voices - 1):
            current += interval
            notes.append(current)

        return VoicingSpec(
            voicing_type=VoicingType.CLUSTER,
            notes=notes,
            root=root,
            chord_tones=[],  # Clusters don't follow traditional chord tones
            extensions=[],
            doublings=[],
            inversion=0,
            spread=0
        )

    @staticmethod
    def generate_shell(root: int, chord_quality: str = "major") -> VoicingSpec:
        """
        Generate shell voicing (root, 3rd, 7th only)

        Minimal voicing for comping. Root in bass, 3rd and 7th in upper voices.
        Essential tones only.

        Args:
            root: Root note
            chord_quality: "major", "minor", "dominant"
        """
        # Define chord tones based on quality
        if chord_quality == "major":
            third = 4  # Major 3rd
            seventh = 11  # Major 7th
        elif chord_quality == "minor":
            third = 3  # Minor 3rd
            seventh = 10  # Minor 7th
        elif chord_quality == "dominant":
            third = 4  # Major 3rd
            seventh = 10  # Minor 7th (dominant 7th)
        else:
            third = 4
            seventh = 10

        notes = [root, root + third, root + seventh]
        notes.sort()

        return VoicingSpec(
            voicing_type=VoicingType.SHELL,
            notes=notes,
            root=root,
            chord_tones=[0, third, seventh],
            extensions=[],
            omissions=[],
            doublings=[],
            inversion=0,
            spread=0
        )


# ============================================================================
# BASS PATTERN GENERATOR
# ============================================================================

class BassPatternGenerator:
    """
    Generates bass line patterns across styles

    Based on techniques from:
    - Ray Brown (walking bass)
    - Ron Carter (modern jazz)
    - Paul Chambers (bebop bass lines)
    - James Jamerson (Motown)
    - Jaco Pastorius (fretless, melodic)
    """

    @staticmethod
    def generate_walking_bass(
        root: int,
        chord_changes: List[Tuple[int, int]],  # [(root1, duration1), (root2, duration2), ...]
        style: str = "swing"
    ) -> List[Tuple[int, float]]:
        """
        Generate walking bass line

        Walking bass: stepwise motion, chord tones on strong beats,
        chromatic approach tones on weak beats.

        Args:
            root: Starting root note
            chord_changes: List of (root_note, duration_in_beats) tuples
            style: "swing", "bebop", "modern"

        Returns:
            List of (note, duration) tuples
        """
        bass_line = []

        for chord_root, duration in chord_changes:
            # Simple walking: root, 3rd, 5th, approach tone
            if duration >= 4:  # Whole note or longer
                # Quarter notes
                notes_per_chord = int(duration)

                # Start on root
                bass_line.append((chord_root, 1.0))

                # Add chord tones and passing tones
                for i in range(1, notes_per_chord):
                    if i % 2 == 0:  # Even beats: chord tones
                        # Alternate 3rd and 5th
                        if i % 4 == 0:
                            bass_line.append((chord_root + 4, 1.0))  # 3rd
                        else:
                            bass_line.append((chord_root + 7, 1.0))  # 5th
                    else:  # Odd beats: approach tones
                        # Chromatic approach to next chord tone
                        next_target = chord_root + (4 if (i+1) % 4 == 0 else 7)
                        approach = next_target - 1  # Chromatic from below
                        bass_line.append((approach, 1.0))
            else:
                # Shorter duration: just root
                bass_line.append((chord_root, duration))

        return bass_line

    @staticmethod
    def generate_pedal(note: int, duration: float) -> List[Tuple[int, float]]:
        """
        Generate pedal tone (sustained bass note)

        Common in:
        - Modal jazz (sustained root or fifth)
        - Classical music (organ points)
        - Rock (drone bass)
        """
        return [(note, duration)]

    @staticmethod
    def generate_ostinato(
        pattern: List[int],
        repetitions: int,
        note_duration: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Generate repeating ostinato pattern

        Args:
            pattern: List of MIDI note numbers
            repetitions: How many times to repeat
            note_duration: Duration of each note
        """
        bass_line = []
        for _ in range(repetitions):
            for note in pattern:
                bass_line.append((note, note_duration))
        return bass_line

    @staticmethod
    def generate_two_feel(
        root: int,
        num_bars: int = 4
    ) -> List[Tuple[int, float]]:
        """
        Generate two-feel bass (half notes, alternating root and fifth)

        Common in:
        - Medium swing tunes (verse sections)
        - Ballads
        - Latin jazz
        """
        bass_line = []
        for bar in range(num_bars):
            # Alternate root and fifth
            if bar % 2 == 0:
                bass_line.append((root, 2.0))
                bass_line.append((root + 7, 2.0))  # Fifth
            else:
                bass_line.append((root + 7, 2.0))
                bass_line.append((root, 2.0))
        return bass_line

    @staticmethod
    def generate_funk_bass(
        root: int,
        pattern_type: str = "basic"
    ) -> List[Tuple[int, float]]:
        """
        Generate funk bass pattern

        Characterized by:
        - Syncopation
        - Ghost notes
        - Octave jumps
        - Sixteenth-note rhythms

        Args:
            root: Root note
            pattern_type: "basic", "slap", "motown"
        """
        if pattern_type == "basic":
            # Classic funk pattern: root on 1, octave on &of2, fifth on 4
            return [
                (root, 0.5),           # 1
                (root, 0.25),          # &of1
                (root - 12, 0.25),     # 2 (octave below)
                (root, 0.25),          # &of2
                (root + 7, 0.25),      # 3 (fifth)
                (root, 0.5),           # &of3
                (root + 7, 0.25),      # 4
                (root, 0.25),          # &of4
            ]
        elif pattern_type == "slap":
            # Slap bass pattern with octaves
            return [
                (root, 0.25),
                (root + 12, 0.25),
                (root, 0.25),
                (root, 0.25),
                (root + 7, 0.25),
                (root, 0.25),
                (root + 12, 0.25),
                (root, 0.25),
            ]
        else:  # motown
            # Motown-style: simple, solid groove
            return [
                (root, 1.0),
                (root + 4, 0.5),
                (root, 0.5),
                (root + 7, 1.0),
                (root, 1.0),
            ]


# ============================================================================
# DRUM PATTERN GENERATOR
# ============================================================================

class DrumPatternGenerator:
    """
    Generates drum patterns across styles

    Based on techniques from:
    - Max Roach (bebop, brush work)
    - Tony Williams (modern jazz, polyrhythm)
    - Elvin Jones (polyrhythmic swing)
    - Steve Gadd (studio precision)
    - Clyde Stubblefield (funk)
    """

    # MIDI drum notes (General MIDI)
    KICK = 36
    SNARE = 38
    CLOSED_HI_HAT = 42
    OPEN_HI_HAT = 46
    RIDE_CYMBAL = 51
    CRASH_CYMBAL = 49

    @staticmethod
    def generate_swing_pattern(feel: str = "medium") -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate swing drum pattern

        Args:
            feel: "slow", "medium", "fast", "uptempo"

        Returns:
            Dict with 'ride', 'hi_hat', 'snare', 'kick' patterns
            Each pattern is list of (beat_position, velocity) tuples
        """
        if feel == "slow":
            # Ballad feel: brushes, soft dynamics
            return {
                'ride': [
                    (0.0, 60), (0.5, 40), (1.0, 60), (1.5, 40),
                    (2.0, 60), (2.5, 40), (3.0, 60), (3.5, 40),
                ],
                'hi_hat': [(0.0, 30), (2.0, 30)],  # 2 and 4
                'snare': [(1.0, 45), (3.0, 45)],  # Backbeat
                'kick': [(0.0, 70), (2.5, 50)],  # Feathering
            }
        elif feel == "medium":
            # Classic swing
            return {
                'ride': [
                    (0.0, 80), (0.667, 50), (1.333, 60),  # Triplet swing
                    (2.0, 80), (2.667, 50), (3.333, 60),
                ],
                'hi_hat': [(1.0, 40), (3.0, 40)],  # 2 and 4
                'snare': [(1.0, 70), (3.0, 70)],
                'kick': [(0.0, 85), (0.5, 50), (2.5, 60)],
            }
        elif feel == "uptempo":
            # Fast bebop
            return {
                'ride': [
                    (0.0, 85), (0.5, 55), (1.0, 85), (1.5, 55),
                    (2.0, 85), (2.5, 55), (3.0, 85), (3.5, 55),
                ],
                'hi_hat': [(1.0, 50), (3.0, 50)],
                'snare': [(1.5, 60), (3.5, 60)],  # Displaced backbeat
                'kick': [(0.0, 80), (2.0, 80)],  # Walking 4
            }
        else:
            return DrumPatternGenerator.generate_swing_pattern("medium")

    @staticmethod
    def generate_rock_pattern(style: str = "basic") -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate rock drum pattern

        Args:
            style: "basic", "shuffle", "half_time"
        """
        if style == "basic":
            # Standard 8th note rock beat
            return {
                'hi_hat': [
                    (0.0, 80), (0.5, 60), (1.0, 80), (1.5, 60),
                    (2.0, 80), (2.5, 60), (3.0, 80), (3.5, 60),
                ],
                'snare': [(1.0, 100), (3.0, 100)],  # 2 and 4
                'kick': [(0.0, 110), (0.5, 80), (2.5, 90), (3.5, 70)],
            }
        elif style == "shuffle":
            # Shuffle rock
            return {
                'hi_hat': [
                    (0.0, 85), (0.667, 55), (1.333, 70),
                    (2.0, 85), (2.667, 55), (3.333, 70),
                ],
                'snare': [(1.0, 105), (3.0, 105)],
                'kick': [(0.0, 115), (2.0, 100), (3.5, 80)],
            }
        else:  # half_time
            # Half-time feel
            return {
                'hi_hat': [
                    (0.0, 75), (0.5, 55), (1.0, 75), (1.5, 55),
                    (2.0, 75), (2.5, 55), (3.0, 75), (3.5, 55),
                ],
                'snare': [(2.0, 110)],  # Only on 3
                'kick': [(0.0, 115), (0.75, 70), (1.5, 80), (3.5, 85)],
            }

    @staticmethod
    def generate_funk_pattern() -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate funk drum pattern (Clyde Stubblefield style)
        """
        return {
            'hi_hat': [
                (0.0, 70), (0.25, 40), (0.5, 85), (0.75, 40),
                (1.0, 70), (1.25, 40), (1.5, 85), (1.75, 40),
                (2.0, 70), (2.25, 40), (2.5, 85), (2.75, 40),
                (3.0, 70), (3.25, 40), (3.5, 85), (3.75, 40),
            ],
            'snare': [
                (1.0, 100),  # 2
                (1.5, 60),   # Ghost note
                (3.0, 105),  # 4
                (3.75, 70),  # &of4
            ],
            'kick': [
                (0.0, 110),
                (0.5, 80),
                (2.25, 90),
                (3.5, 85),
            ],
        }

    @staticmethod
    def generate_latin_pattern(style: str = "bossa") -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate Latin drum patterns

        Args:
            style: "bossa", "samba", "afro_cuban"
        """
        if style == "bossa":
            # Bossa nova pattern
            return {
                'ride': [  # Cross-stick or rim
                    (0.0, 65), (0.5, 50), (1.5, 60),
                    (2.0, 65), (3.0, 60), (3.5, 50),
                ],
                'kick': [(0.0, 85), (0.5, 70), (2.5, 80)],
                'snare': [(1.0, 55), (2.0, 60)],  # Light, crisp
            }
        elif style == "samba":
            # Samba pattern
            return {
                'ride': [  # 16th notes
                    (0.0, 70), (0.25, 50), (0.5, 70), (0.75, 50),
                    (1.0, 70), (1.25, 50), (1.5, 70), (1.75, 50),
                ],
                'kick': [(0.0, 90), (1.0, 85), (1.5, 80)],
                'snare': [(0.5, 75), (1.25, 70), (1.75, 65)],
            }
        else:  # afro_cuban
            # Afro-Cuban (son clave based)
            return {
                'ride': [
                    (0.0, 75), (0.5, 55), (1.0, 75), (1.5, 55),
                    (2.0, 75), (2.5, 55), (3.0, 75), (3.5, 55),
                ],
                'kick': [(0.0, 90), (2.5, 85)],  # Tumbao
                'snare': [(1.5, 70), (3.0, 75)],
            }


# ============================================================================
# BRASS SECTION VOICING
# ============================================================================

class BrassSectionVoicing:
    """
    Brass section voicing techniques

    Based on:
    - Duke Ellington (characteristic voicings)
    - Count Basie (clean, crisp sections)
    - Thad Jones (modern big band)
    - Sammy Nestico (educational, clear)
    - Maria Schneider (contemporary)
    """

    @staticmethod
    def four_way_close(
        root: int,
        chord_tones: List[int],
        lead_note: int = None
    ) -> Dict[str, int]:
        """
        Four-way close voicing for brass section

        Typically: Trumpet 1, Trumpet 2, Trombone 1, Trombone 2
        All voices within an octave of the lead

        Args:
            root: Root note of chord
            chord_tones: Intervals from root [0, 4, 7, 10] for dom7
            lead_note: Top note (if None, use highest chord tone)

        Returns:
            Dict mapping instrument to MIDI note
        """
        if lead_note is None:
            lead_note = root + max(chord_tones)

        # Build close voicing below lead
        voicing = [lead_note]
        remaining_tones = sorted([root + ct for ct in chord_tones if root + ct < lead_note], reverse=True)

        for tone in remaining_tones[:3]:  # Take 3 notes below lead
            voicing.append(tone)

        # Ensure 4 voices
        while len(voicing) < 4:
            voicing.append(voicing[-1] - 3)  # Add thirds below

        voicing.sort(reverse=True)

        return {
            'Trumpet 1': voicing[0],
            'Trumpet 2': voicing[1],
            'Trombone 1': voicing[2],
            'Trombone 2': voicing[3],
        }

    @staticmethod
    def four_way_open(
        root: int,
        chord_tones: List[int]
    ) -> Dict[str, int]:
        """
        Four-way open (spread) voicing

        More space between voices, fuller sound
        """
        notes = [root + ct for ct in sorted(chord_tones)]

        # Spread voicing over wider range
        spread_voicing = [
            notes[3] if len(notes) > 3 else notes[-1],  # Lead
            notes[2] if len(notes) > 2 else notes[-1] - 4,
            notes[1] if len(notes) > 1 else notes[-1] - 7,
            notes[0] - 12,  # Bass note an octave lower
        ]

        spread_voicing.sort(reverse=True)

        return {
            'Trumpet 1': spread_voicing[0],
            'Trumpet 2': spread_voicing[1],
            'Trombone 1': spread_voicing[2],
            'Trombone 2': spread_voicing[3],
        }

    @staticmethod
    def five_way_close(
        root: int,
        chord_tones: List[int],
        extensions: List[int] = None
    ) -> Dict[str, int]:
        """
        Five-way close voicing (full big band brass)

        4 Trumpets + 4 Trombones = 5-part harmony
        (Often doubles some parts)
        """
        if extensions is None:
            extensions = []

        all_tones = sorted(set(chord_tones + extensions))
        notes = [root + t for t in all_tones]

        # Take top 5 notes
        while len(notes) < 5:
            notes.append(notes[-1] + 3)

        voicing = sorted(notes[-5:], reverse=True)

        return {
            'Trumpet 1': voicing[0],
            'Trumpet 2': voicing[1],
            'Trumpet 3': voicing[2],
            'Trombone 1': voicing[3],
            'Trombone 2': voicing[4],
        }

    @staticmethod
    def double_lead(
        root: int,
        chord_tones: List[int],
        lead_note: int
    ) -> Dict[str, int]:
        """
        Double lead voicing (2 trumpets on melody)

        Creates powerful, bright lead sound
        Basie and Ellington used this frequently
        """
        voicing_dict = BrassSectionVoicing.four_way_close(root, chord_tones, lead_note)

        # Add doubled lead
        voicing_dict['Trumpet 1 (double)'] = voicing_dict['Trumpet 1']

        return voicing_dict


# ============================================================================
# INSTRUMENTATION SPECIALIST (MAIN CLASS)
# ============================================================================

class InstrumentationSpecialist:
    """
    Main instrumentation specialist engine

    Coordinates all instrumentation decisions:
    - Instrument selection
    - Orchestration
    - Voicing assignment
    - Balance and blend
    - Articulation
    """

    def __init__(self):
        self.knowledge = OrchestrationKnowledge()
        self.piano_voicings = PianoVoicingGenerator()
        self.bass_patterns = BassPatternGenerator()
        self.drum_patterns = DrumPatternGenerator()
        self.brass_voicings = BrassSectionVoicing()

        # Initialize ensemble templates
        self.ensemble_templates = self._create_ensemble_templates()

    def _create_ensemble_templates(self) -> Dict[str, EnsembleTemplate]:
        """
        Create standard ensemble templates
        """
        templates = {}

        # Jazz combo
        templates['jazz_trio'] = EnsembleTemplate(
            name="Jazz Trio",
            ensemble_type=EnsembleType.TRIO,
            instruments=["Piano", "Double Bass", "Drums"],
            typical_roles={
                "Piano": InstrumentRole.HARMONY,
                "Double Bass": InstrumentRole.BASS,
                "Drums": InstrumentRole.RHYTHM,
            },
            voicing_preferences=[VoicingType.ROOTLESS, VoicingType.DROP_2, VoicingType.SHELL],
            genre_associations=["jazz", "bebop", "swing"],
            texture_range=(TextureDensity.LIGHT, TextureDensity.MEDIUM)
        )

        templates['jazz_quartet'] = EnsembleTemplate(
            name="Jazz Quartet",
            ensemble_type=EnsembleType.QUARTET,
            instruments=["Trumpet", "Tenor Saxophone", "Piano", "Double Bass", "Drums"],
            typical_roles={
                "Trumpet": InstrumentRole.MELODY,
                "Tenor Saxophone": InstrumentRole.MELODY,
                "Piano": InstrumentRole.HARMONY,
                "Double Bass": InstrumentRole.BASS,
                "Drums": InstrumentRole.RHYTHM,
            },
            voicing_preferences=[VoicingType.ROOTLESS, VoicingType.DROP_2],
            genre_associations=["jazz", "bebop", "hard_bop"],
            texture_range=(TextureDensity.MEDIUM, TextureDensity.FULL)
        )

        # Big band
        templates['big_band'] = EnsembleTemplate(
            name="Big Band",
            ensemble_type=EnsembleType.BIG_BAND,
            instruments=[
                "Trumpet", "Trumpet", "Trumpet", "Trumpet",
                "Alto Saxophone", "Alto Saxophone", "Tenor Saxophone", "Tenor Saxophone",
                "Baritone Saxophone",
                "Trombone", "Trombone", "Trombone", "Bass Trombone",
                "Piano", "Double Bass", "Drums"
            ],
            typical_roles={
                "Trumpet": InstrumentRole.MELODY,
                "Alto Saxophone": InstrumentRole.HARMONY,
                "Tenor Saxophone": InstrumentRole.HARMONY,
                "Baritone Saxophone": InstrumentRole.BASS,
                "Trombone": InstrumentRole.HARMONY,
                "Piano": InstrumentRole.HARMONY,
                "Double Bass": InstrumentRole.BASS,
                "Drums": InstrumentRole.RHYTHM,
            },
            voicing_preferences=[
                VoicingType.FOUR_WAY_CLOSE,
                VoicingType.FIVE_WAY_CLOSE,
                VoicingType.DROP_2
            ],
            genre_associations=["jazz", "swing", "big_band"],
            texture_range=(TextureDensity.FULL, TextureDensity.DENSE)
        )

        # Classical chamber
        templates['string_quartet'] = EnsembleTemplate(
            name="String Quartet",
            ensemble_type=EnsembleType.STRING_QUARTET,
            instruments=["Violin", "Violin", "Viola", "Cello"],
            typical_roles={
                "Violin": InstrumentRole.MELODY,
                "Viola": InstrumentRole.HARMONY,
                "Cello": InstrumentRole.BASS,
            },
            voicing_preferences=[VoicingType.CLOSE_POSITION, VoicingType.DIVISI],
            genre_associations=["classical", "chamber"],
            texture_range=(TextureDensity.LIGHT, TextureDensity.MEDIUM)
        )

        templates['brass_quintet'] = EnsembleTemplate(
            name="Brass Quintet",
            ensemble_type=EnsembleType.BRASS_QUINTET,
            instruments=["Trumpet", "Trumpet", "Horn", "Trombone", "Tuba"],
            typical_roles={
                "Trumpet": InstrumentRole.MELODY,
                "Horn": InstrumentRole.HARMONY,
                "Trombone": InstrumentRole.HARMONY,
                "Tuba": InstrumentRole.BASS,
            },
            voicing_preferences=[VoicingType.CLOSE_POSITION, VoicingType.SPREAD_POSITION],
            genre_associations=["classical", "chamber", "brass"],
            texture_range=(TextureDensity.MEDIUM, TextureDensity.FULL)
        )

        templates['woodwind_quintet'] = EnsembleTemplate(
            name="Woodwind Quintet",
            ensemble_type=EnsembleType.WOODWIND_QUINTET,
            instruments=["Flute", "Oboe", "Clarinet", "Horn", "Bassoon"],
            typical_roles={
                "Flute": InstrumentRole.MELODY,
                "Oboe": InstrumentRole.MELODY,
                "Clarinet": InstrumentRole.HARMONY,
                "Horn": InstrumentRole.HARMONY,
                "Bassoon": InstrumentRole.BASS,
            },
            voicing_preferences=[VoicingType.CLOSE_POSITION, VoicingType.SPREAD_POSITION],
            genre_associations=["classical", "chamber", "wind"],
            texture_range=(TextureDensity.LIGHT, TextureDensity.MEDIUM)
        )

        # Rock/Pop
        templates['rock_band'] = EnsembleTemplate(
            name="Rock Band",
            ensemble_type=EnsembleType.ROCK_BAND,
            instruments=["Electric Guitar", "Electric Bass", "Drums", "Vocals"],
            typical_roles={
                "Electric Guitar": InstrumentRole.HARMONY,
                "Electric Bass": InstrumentRole.BASS,
                "Drums": InstrumentRole.RHYTHM,
                "Vocals": InstrumentRole.MELODY,
            },
            voicing_preferences=[VoicingType.ROOT_POSITION, VoicingType.DROP_2],
            genre_associations=["rock", "pop"],
            texture_range=(TextureDensity.MEDIUM, TextureDensity.FULL)
        )

        return templates

    def select_ensemble(
        self,
        genre: str,
        texture_density: float = 0.7,
        custom_instruments: Optional[List[str]] = None
    ) -> InstrumentationProfile:
        """
        Select appropriate ensemble for genre and texture

        Args:
            genre: Musical genre ("jazz", "classical", "rock", etc.)
            texture_density: 0.0 (sparse) to 1.0 (dense)
            custom_instruments: Optional list of specific instruments

        Returns:
            InstrumentationProfile with selected ensemble
        """
        # Map texture density to TextureDensity enum
        if texture_density < 0.2:
            density_enum = TextureDensity.SPARSE
        elif texture_density < 0.4:
            density_enum = TextureDensity.LIGHT
        elif texture_density < 0.6:
            density_enum = TextureDensity.MEDIUM
        elif texture_density < 0.8:
            density_enum = TextureDensity.FULL
        else:
            density_enum = TextureDensity.DENSE

        # Find matching templates
        matching_templates = [
            tmpl for tmpl in self.ensemble_templates.values()
            if genre in tmpl.genre_associations
        ]

        if not matching_templates:
            # Default to jazz trio
            template = self.ensemble_templates['jazz_trio']
        else:
            # Pick template matching density
            template = matching_templates[0]
            for tmpl in matching_templates:
                min_density, max_density = tmpl.texture_range
                # Simple comparison by enum value
                if min_density.value <= density_enum.value <= max_density.value:
                    template = tmpl
                    break

        # Get instrument objects
        if custom_instruments:
            instruments = [get_instrument(name) for name in custom_instruments]
            instruments = [inst for inst in instruments if inst is not None]
        else:
            instruments = [get_instrument(name) for name in template.instruments]
            instruments = [inst for inst in instruments if inst is not None]

        # Determine ensemble type
        ensemble_type = template.ensemble_type

        # Create profile
        profile = InstrumentationProfile(
            ensemble_type=ensemble_type,
            instruments=instruments,
            roles={inst.name: template.typical_roles.get(inst.name, InstrumentRole.HARMONY)
                   for inst in instruments},
            voicing_type=template.voicing_preferences[0] if template.voicing_preferences else VoicingType.DROP_2,
            texture_density=density_enum,
        )

        # Calculate blend score
        profile.blend_score = self.calculate_blend_score(instruments)

        return profile

    def calculate_blend_score(self, instruments: List[Instrument]) -> float:
        """
        Calculate blend compatibility score for ensemble

        Args:
            instruments: List of Instrument objects

        Returns:
            Blend score from 0.0 (poor) to 1.0 (excellent)
        """
        if len(instruments) < 2:
            return 1.0  # Solo always blends perfectly

        total_score = 0.0
        comparisons = 0

        # Check all pairwise combinations
        for i, inst_a in enumerate(instruments):
            for inst_b in instruments[i+1:]:
                # Find blend data
                blend_data = None
                for blend in self.knowledge.BLEND_COMPATIBILITY_DB:
                    if ((blend.instrument_a == inst_a.name and blend.instrument_b == inst_b.name) or
                        (blend.instrument_a == inst_b.name and blend.instrument_b == inst_a.name)):
                        blend_data = blend
                        break

                if blend_data:
                    total_score += blend_data.compatibility_score
                else:
                    # Default: same family blends well, different families less so
                    if inst_a.family == inst_b.family:
                        total_score += 0.8
                    else:
                        total_score += 0.5

                comparisons += 1

        return total_score / comparisons if comparisons > 0 else 1.0

    def assign_voicing(
        self,
        profile: InstrumentationProfile,
        root: int,
        chord_tones: List[int],
        extensions: List[int] = None
    ) -> Dict[str, List[int]]:
        """
        Assign specific voicing to ensemble

        Args:
            profile: InstrumentationProfile
            root: Root note of chord
            chord_tones: Chord tone intervals
            extensions: Extension intervals

        Returns:
            Dict mapping instrument name to list of MIDI notes
        """
        if extensions is None:
            extensions = []

        voicing_assignments = {}

        # Check if we have piano
        piano_instruments = [inst for inst in profile.instruments if "Piano" in inst.name]
        if piano_instruments and profile.voicing_type in [
            VoicingType.DROP_2, VoicingType.DROP_3, VoicingType.ROOTLESS,
            VoicingType.QUARTAL, VoicingType.CLUSTER, VoicingType.SHELL
        ]:
            # Generate piano voicing
            if profile.voicing_type == VoicingType.DROP_2:
                voicing = self.piano_voicings.generate_drop_2(root, chord_tones, extensions)
            elif profile.voicing_type == VoicingType.DROP_3:
                voicing = self.piano_voicings.generate_drop_3(root, chord_tones, extensions)
            elif profile.voicing_type == VoicingType.ROOTLESS:
                voicing = self.piano_voicings.generate_rootless(root, chord_tones, extensions)
            elif profile.voicing_type == VoicingType.QUARTAL:
                voicing = self.piano_voicings.generate_quartal(root, 4)
            elif profile.voicing_type == VoicingType.CLUSTER:
                voicing = self.piano_voicings.generate_cluster(root, 4)
            elif profile.voicing_type == VoicingType.SHELL:
                voicing = self.piano_voicings.generate_shell(root, "dominant")

            voicing_assignments[piano_instruments[0].name] = voicing.notes

        # Check if we have brass section
        brass = [inst for inst in profile.instruments if inst.family == InstrumentFamily.BRASS]
        if len(brass) >= 4 and profile.voicing_type in [
            VoicingType.FOUR_WAY_CLOSE, VoicingType.FOUR_WAY_OPEN,
            VoicingType.FIVE_WAY_CLOSE, VoicingType.DOUBLE_LEAD
        ]:
            # Generate brass voicing
            if profile.voicing_type == VoicingType.FOUR_WAY_CLOSE:
                brass_voicing = self.brass_voicings.four_way_close(root, chord_tones)
            elif profile.voicing_type == VoicingType.FOUR_WAY_OPEN:
                brass_voicing = self.brass_voicings.four_way_open(root, chord_tones)
            elif profile.voicing_type == VoicingType.FIVE_WAY_CLOSE:
                brass_voicing = self.brass_voicings.five_way_close(root, chord_tones, extensions)
            elif profile.voicing_type == VoicingType.DOUBLE_LEAD:
                brass_voicing = self.brass_voicings.double_lead(root, chord_tones, root + max(chord_tones))

            # Assign to actual instruments
            for inst_name, note in brass_voicing.items():
                # Find matching instrument
                matching = [inst for inst in brass if inst_name.split()[0] in inst.name]
                if matching:
                    voicing_assignments[matching[0].name] = [note]

        return voicing_assignments

    def recommend_doublings(
        self,
        profile: InstrumentationProfile,
        melody_instrument: str
    ) -> Dict[str, List[str]]:
        """
        Recommend doubling instruments for melody

        Args:
            profile: InstrumentationProfile
            melody_instrument: Primary melody instrument name

        Returns:
            Dict with doubling recommendations
        """
        recommendations = {}

        melody_inst = get_instrument(melody_instrument)
        if not melody_inst:
            return recommendations

        available = [inst for inst in profile.instruments if inst.name != melody_instrument]

        # Check blend compatibility
        good_doubles = []
        for inst in available:
            # Find blend score
            blend_score = 0.5  # Default
            for blend in self.knowledge.BLEND_COMPATIBILITY_DB:
                if ((blend.instrument_a == melody_inst.name and blend.instrument_b == inst.name) or
                    (blend.instrument_a == inst.name and blend.instrument_b == melody_inst.name)):
                    blend_score = blend.compatibility_score
                    break

            if blend_score >= 0.7:
                good_doubles.append(inst.name)

        recommendations['unison'] = good_doubles[:2] if len(good_doubles) >= 2 else good_doubles
        recommendations['octave'] = good_doubles[2:4] if len(good_doubles) >= 4 else []

        return recommendations

    def assign_articulations(
        self,
        profile: InstrumentationProfile,
        musical_context: str = "legato"
    ) -> Dict[str, ArticulationType]:
        """
        Assign appropriate articulations to instruments

        Args:
            profile: InstrumentationProfile
            musical_context: "legato", "staccato", "marcato", "swing"

        Returns:
            Dict mapping instrument name to ArticulationType
        """
        assignments = {}

        for inst in profile.instruments:
            # Default articulation based on context and family
            if musical_context == "legato":
                if inst.family == InstrumentFamily.STRINGS:
                    assignments[inst.name] = ArticulationType.ARCO
                elif inst.family == InstrumentFamily.BRASS:
                    assignments[inst.name] = ArticulationType.LEGATO
                elif inst.family == InstrumentFamily.WOODWINDS:
                    assignments[inst.name] = ArticulationType.TONGUED
                else:
                    assignments[inst.name] = ArticulationType.LEGATO

            elif musical_context == "staccato":
                if inst.family == InstrumentFamily.STRINGS:
                    assignments[inst.name] = ArticulationType.SPICCATO
                else:
                    assignments[inst.name] = ArticulationType.STACCATO

            elif musical_context == "marcato":
                assignments[inst.name] = ArticulationType.MARCATO

            elif musical_context == "swing":
                if inst.family == InstrumentFamily.BRASS:
                    # Brass in swing: generally straight, clean attacks
                    assignments[inst.name] = ArticulationType.STRAIGHT
                elif inst.family == InstrumentFamily.WOODWINDS:
                    assignments[inst.name] = ArticulationType.TONGUED
                else:
                    assignments[inst.name] = ArticulationType.LEGATO

        return assignments

    def generate_bass_line(
        self,
        pattern_type: BassPattern,
        chord_root: int,
        duration: float = 4.0,
        style: str = "swing"
    ) -> List[Tuple[int, float]]:
        """
        Generate bass line pattern

        Args:
            pattern_type: BassPattern enum
            chord_root: Root note
            duration: Duration in beats
            style: Style modifier

        Returns:
            List of (note, duration) tuples
        """
        if pattern_type == BassPattern.WALKING:
            return self.bass_patterns.generate_walking_bass(
                chord_root,
                [(chord_root, duration)],
                style
            )
        elif pattern_type == BassPattern.PEDAL:
            return self.bass_patterns.generate_pedal(chord_root, duration)
        elif pattern_type == BassPattern.TWO_FEEL:
            num_bars = int(duration / 4)
            return self.bass_patterns.generate_two_feel(chord_root, num_bars)
        elif pattern_type == BassPattern.FUNK:
            return self.bass_patterns.generate_funk_bass(chord_root, "basic")
        else:
            # Default: whole note
            return [(chord_root, duration)]

    def generate_drum_pattern(
        self,
        pattern_type: DrumPattern,
        feel: str = "medium"
    ) -> Dict[str, List[Tuple[float, int]]]:
        """
        Generate drum pattern

        Args:
            pattern_type: DrumPattern enum
            feel: Feel modifier ("slow", "medium", "fast", etc.)

        Returns:
            Dict with drum voice patterns
        """
        if pattern_type == DrumPattern.SWING:
            return self.drum_patterns.generate_swing_pattern(feel)
        elif pattern_type == DrumPattern.ROCK_BASIC:
            return self.drum_patterns.generate_rock_pattern("basic")
        elif pattern_type == DrumPattern.ROCK_SHUFFLE:
            return self.drum_patterns.generate_rock_pattern("shuffle")
        elif pattern_type == DrumPattern.FUNK:
            return self.drum_patterns.generate_funk_pattern()
        elif pattern_type == DrumPattern.BOSSA_NOVA:
            return self.drum_patterns.generate_latin_pattern("bossa")
        elif pattern_type == DrumPattern.SAMBA:
            return self.drum_patterns.generate_latin_pattern("samba")
        elif pattern_type == DrumPattern.AFRO_CUBAN:
            return self.drum_patterns.generate_latin_pattern("afro_cuban")
        else:
            return self.drum_patterns.generate_swing_pattern(feel)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_jazz_trio() -> InstrumentationProfile:
    """Quick function to create standard jazz trio"""
    specialist = InstrumentationSpecialist()
    return specialist.select_ensemble("jazz", 0.5)


def create_big_band() -> InstrumentationProfile:
    """Quick function to create big band ensemble"""
    specialist = InstrumentationSpecialist()
    return specialist.select_ensemble("jazz", 0.9)


def create_string_quartet() -> InstrumentationProfile:
    """Quick function to create string quartet"""
    specialist = InstrumentationSpecialist()
    return specialist.select_ensemble("classical", 0.5,
                                     ["Violin", "Violin", "Viola", "Cello"])


def demo_piano_voicings():
    """
    Demonstrate piano voicing generation
    """
    print("=" * 80)
    print("PIANO VOICING DEMONSTRATION")
    print("=" * 80)

    generator = PianoVoicingGenerator()
    root = 60  # Middle C

    # Dominant 7th chord: C E G Bb
    chord_tones = [0, 4, 7, 10]
    extensions = [14]  # 9th

    print(f"\nChord: C7(9) (Root: {root})")
    print(f"Chord Tones: {chord_tones}")
    print(f"Extensions: {extensions}\n")

    # Drop-2
    drop2 = generator.generate_drop_2(root, chord_tones, extensions)
    print(f"Drop-2 Voicing: {drop2.notes}")
    print(f"  Notes: {[note - root for note in drop2.notes]}")

    # Drop-3
    drop3 = generator.generate_drop_3(root, chord_tones, extensions)
    print(f"Drop-3 Voicing: {drop3.notes}")
    print(f"  Notes: {[note - root for note in drop3.notes]}")

    # Rootless
    rootless = generator.generate_rootless(root, chord_tones, extensions)
    print(f"Rootless Voicing: {rootless.notes}")
    print(f"  Notes: {[note - root for note in rootless.notes]}")

    # Quartal
    quartal = generator.generate_quartal(root, 4)
    print(f"Quartal Voicing: {quartal.notes}")
    print(f"  Notes: {[note - root for note in quartal.notes]}")

    # Shell
    shell = generator.generate_shell(root, "dominant")
    print(f"Shell Voicing: {shell.notes}")
    print(f"  Notes: {[note - root for note in shell.notes]}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Quick demo
    print("Agent 21: Instrumentation Specialist - Loaded Successfully")
    print(f"Total lines: {sum(1 for _ in open(__file__))}")
    demo_piano_voicings()
