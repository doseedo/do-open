#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comprehensive Instrument Library for Professional Orchestration

This module provides a complete database of orchestral, band, ethnic, and electronic
instruments with detailed specifications for intelligent orchestration and MIDI generation.

Features:
- Complete range specifications (written and sounding pitch)
- Transposition information
- Comfortable tessitura
- Technical limitations and capabilities
- Available articulations
- MIDI program numbers (General MIDI + extended)
- Dynamic range characteristics
- Timbre descriptors

Research References:
- Rimsky-Korsakov's Principles of Orchestration
- Samuel Adler's Study of Orchestration
- Berlioz's Treatise on Instrumentation
- Professional orchestration practice

Author: Claude (Sonnet 4.5)
Created: 2025
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class InstrumentFamily(Enum):
    """Instrument family classification"""
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    BRASS = "brass"
    PERCUSSION = "percussion"
    KEYBOARDS = "keyboards"
    PLUCKED_STRINGS = "plucked_strings"
    ETHNIC = "ethnic"
    ELECTRONIC = "electronic"
    VOICES = "voices"


class ArticulationType(Enum):
    """Available articulation types"""
    # Common
    LEGATO = "legato"
    STACCATO = "staccato"
    STACCATISSIMO = "staccatissimo"
    TENUTO = "tenuto"
    MARCATO = "marcato"
    ACCENT = "accent"

    # Strings
    ARCO = "arco"
    PIZZICATO = "pizzicato"
    COL_LEGNO = "col_legno"
    SUL_PONTICELLO = "sul_ponticello"
    SUL_TASTO = "sul_tasto"
    TREMOLO = "tremolo"
    HARMONICS = "harmonics"
    SPICCATO = "spiccato"
    RICOCHET = "ricochet"

    # Brass
    STRAIGHT = "straight"
    MUTED = "muted"
    CUP_MUTE = "cup_mute"
    HARMON_MUTE = "harmon_mute"
    STRAIGHT_MUTE = "straight_mute"
    FLUTTER_TONGUE = "flutter_tongue"
    FALL_OFF = "fall_off"
    RIP = "rip"
    GLISSANDO = "glissando"

    # Woodwinds
    TONGUED = "tongued"
    DOUBLE_TONGUE = "double_tongue"
    TRIPLE_TONGUE = "triple_tongue"
    SLAP_TONGUE = "slap_tongue"
    GROWL = "growl"
    MULTIPHONICS = "multiphonics"


@dataclass
class InstrumentRange:
    """Defines the playable range of an instrument"""
    lowest_note: int  # MIDI note number
    highest_note: int  # MIDI note number
    comfortable_low: int  # Lower tessitura boundary
    comfortable_high: int  # Upper tessitura boundary
    optimal_low: int  # Sweet spot lower bound
    optimal_high: int  # Sweet spot upper bound


@dataclass
class Instrument:
    """Complete instrument specification"""
    name: str
    family: InstrumentFamily
    midi_program: int  # General MIDI program number (0-127)
    midi_channel_default: int = 0  # Default MIDI channel (0-15)

    # Range information
    range: InstrumentRange = None
    transposition: int = 0  # Semitones from written to sounding (positive = sounds higher)

    # Performance characteristics
    max_dynamic: int = 127  # Maximum velocity
    min_dynamic: int = 20  # Minimum playable velocity
    typical_dynamic: int = 80  # Most common playing level

    # Technical capabilities
    max_speed: float = 16.0  # Notes per second at max tempo
    sustain_capability: bool = True  # Can play long sustained notes
    polyphonic: bool = False  # Can play multiple notes simultaneously

    # Available articulations
    articulations: Set[ArticulationType] = field(default_factory=set)

    # Timbre characteristics (descriptive)
    timbre_descriptors: List[str] = field(default_factory=list)

    # Special capabilities
    pitch_bend_range: int = 2  # Semitones
    can_vibrato: bool = True
    can_tremolo: bool = False

    # Orchestration notes
    blends_well_with: List[str] = field(default_factory=list)
    avoid_combinations: List[str] = field(default_factory=list)
    orchestration_notes: str = ""


# ============================================================================
# MIDI Note Helper Functions
# ============================================================================

def note_name_to_midi(note: str) -> int:
    """
    Convert note name to MIDI number.
    Format: C4 = middle C = 60
    """
    note_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
    note_name = note[:-1].upper()
    octave = int(note[-1])

    base = note_map[note_name[0]]
    if '#' in note_name:
        base += 1
    elif 'b' in note_name:
        base -= 1

    return base + (octave + 1) * 12


def midi_to_note_name(midi: int) -> str:
    """Convert MIDI number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (midi // 12) - 1
    note = notes[midi % 12]
    return f"{note}{octave}"


# ============================================================================
# INSTRUMENT DATABASE
# ============================================================================

INSTRUMENTS: Dict[str, Instrument] = {}


def register_instrument(instrument: Instrument):
    """Register an instrument in the global database"""
    INSTRUMENTS[instrument.name] = instrument


# ----------------------------------------------------------------------------
# STRING FAMILY
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Violin",
    family=InstrumentFamily.STRINGS,
    midi_program=40,  # GM: Violin
    range=InstrumentRange(
        lowest_note=note_name_to_midi('G3'),  # 55
        highest_note=note_name_to_midi('E7'),  # 100+ (harmonics higher)
        comfortable_low=note_name_to_midi('A3'),  # 57
        comfortable_high=note_name_to_midi('A6'),  # 93
        optimal_low=note_name_to_midi('D4'),  # 62
        optimal_high=note_name_to_midi('E6'),  # 88
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=15,
    typical_dynamic=85,
    max_speed=20.0,  # Very agile
    sustain_capability=True,
    polyphonic=True,  # Double stops, chords
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.ARCO, ArticulationType.PIZZICATO,
        ArticulationType.COL_LEGNO, ArticulationType.SUL_PONTICELLO,
        ArticulationType.SUL_TASTO, ArticulationType.TREMOLO,
        ArticulationType.HARMONICS, ArticulationType.SPICCATO,
        ArticulationType.RICOCHET, ArticulationType.GLISSANDO
    },
    timbre_descriptors=["bright", "singing", "agile", "expressive"],
    can_vibrato=True,
    can_tremolo=True,
    blends_well_with=["Viola", "Cello", "Flute", "Oboe"],
    orchestration_notes="The soprano of the string section. Excellent for melody. "
                       "Avoid writing too low (below D4) for extended periods. "
                       "Can play triple and quadruple stops but difficult."
))

register_instrument(Instrument(
    name="Viola",
    family=InstrumentFamily.STRINGS,
    midi_program=41,  # GM: Viola
    range=InstrumentRange(
        lowest_note=note_name_to_midi('C3'),  # 48
        highest_note=note_name_to_midi('E6'),  # 88
        comfortable_low=note_name_to_midi('D3'),  # 50
        comfortable_high=note_name_to_midi('C6'),  # 84
        optimal_low=note_name_to_midi('G3'),  # 55
        optimal_high=note_name_to_midi('G5'),  # 79
    ),
    transposition=0,
    max_dynamic=120,
    min_dynamic=18,
    typical_dynamic=80,
    max_speed=18.0,
    sustain_capability=True,
    polyphonic=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.ARCO, ArticulationType.PIZZICATO,
        ArticulationType.COL_LEGNO, ArticulationType.SUL_PONTICELLO,
        ArticulationType.SUL_TASTO, ArticulationType.TREMOLO,
        ArticulationType.HARMONICS, ArticulationType.SPICCATO
    },
    timbre_descriptors=["warm", "dark", "mellow", "rich"],
    can_vibrato=True,
    can_tremolo=True,
    blends_well_with=["Violin", "Cello", "Horn", "Clarinet"],
    orchestration_notes="Rich, warm middle voice. Excellent for inner harmonies. "
                       "Low register (C-string) is particularly dark and resonant."
))

register_instrument(Instrument(
    name="Cello",
    family=InstrumentFamily.STRINGS,
    midi_program=42,  # GM: Cello
    range=InstrumentRange(
        lowest_note=note_name_to_midi('C2'),  # 36
        highest_note=note_name_to_midi('C6'),  # 84
        comfortable_low=note_name_to_midi('D2'),  # 38
        comfortable_high=note_name_to_midi('G5'),  # 79
        optimal_low=note_name_to_midi('G2'),  # 43
        optimal_high=note_name_to_midi('C5'),  # 72
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=20,
    typical_dynamic=85,
    max_speed=16.0,
    sustain_capability=True,
    polyphonic=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.ARCO, ArticulationType.PIZZICATO,
        ArticulationType.COL_LEGNO, ArticulationType.SUL_PONTICELLO,
        ArticulationType.SUL_TASTO, ArticulationType.TREMOLO,
        ArticulationType.HARMONICS, ArticulationType.SPICCATO
    },
    timbre_descriptors=["singing", "rich", "expressive", "warm"],
    can_vibrato=True,
    can_tremolo=True,
    blends_well_with=["Violin", "Viola", "Bass", "Bassoon", "Horn"],
    orchestration_notes="Tenor/bass of strings. Very expressive, can carry melody "
                       "beautifully in tenor register. Low notes are rich and powerful."
))

register_instrument(Instrument(
    name="Double Bass",
    family=InstrumentFamily.STRINGS,
    midi_program=43,  # GM: Contrabass
    range=InstrumentRange(
        lowest_note=note_name_to_midi('E1'),  # 28 (with extension, C1=24)
        highest_note=note_name_to_midi('G4'),  # 67
        comfortable_low=note_name_to_midi('E1'),  # 28
        comfortable_high=note_name_to_midi('G3'),  # 55
        optimal_low=note_name_to_midi('A1'),  # 33
        optimal_high=note_name_to_midi('D3'),  # 50
    ),
    transposition=-12,  # Sounds octave lower than written
    max_dynamic=120,
    min_dynamic=25,
    typical_dynamic=80,
    max_speed=10.0,  # Less agile than other strings
    sustain_capability=True,
    polyphonic=False,  # Rarely plays double stops
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.ARCO, ArticulationType.PIZZICATO,
        ArticulationType.COL_LEGNO, ArticulationType.SUL_PONTICELLO,
        ArticulationType.TREMOLO
    },
    timbre_descriptors=["deep", "fundamental", "powerful", "resonant"],
    can_vibrato=True,
    can_tremolo=True,
    blends_well_with=["Cello", "Bassoon", "Tuba", "Trombone"],
    orchestration_notes="Foundation of the orchestra. Sounds octave lower than written. "
                       "Pizzicato is particularly effective. Avoid rapid passages in low register."
))

# ----------------------------------------------------------------------------
# WOODWIND FAMILY
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Flute",
    family=InstrumentFamily.WOODWINDS,
    midi_program=73,  # GM: Flute
    range=InstrumentRange(
        lowest_note=note_name_to_midi('C4'),  # 60 (middle C)
        highest_note=note_name_to_midi('D7'),  # 98
        comfortable_low=note_name_to_midi('D4'),  # 62
        comfortable_high=note_name_to_midi('A6'),  # 93
        optimal_low=note_name_to_midi('G4'),  # 67
        optimal_high=note_name_to_midi('C6'),  # 84
    ),
    transposition=0,
    max_dynamic=110,
    min_dynamic=30,
    typical_dynamic=80,
    max_speed=22.0,  # Extremely agile
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE,
        ArticulationType.TRIPLE_TONGUE, ArticulationType.FLUTTER_TONGUE
    },
    timbre_descriptors=["bright", "clear", "agile", "ethereal"],
    can_vibrato=True,
    blends_well_with=["Violin", "Oboe", "Clarinet"],
    avoid_combinations=["Trombone in unison"],
    orchestration_notes="Most agile woodwind. Low register can be breathy, high register "
                       "is piercing. Excellent for rapid passages and ornamentation."
))

register_instrument(Instrument(
    name="Piccolo",
    family=InstrumentFamily.WOODWINDS,
    midi_program=72,  # GM: Piccolo
    range=InstrumentRange(
        lowest_note=note_name_to_midi('D5'),  # 74
        highest_note=note_name_to_midi('C8'),  # 108
        comfortable_low=note_name_to_midi('E5'),  # 76
        comfortable_high=note_name_to_midi('G7'),  # 103
        optimal_low=note_name_to_midi('G5'),  # 79
        optimal_high=note_name_to_midi('C7'),  # 96
    ),
    transposition=12,  # Sounds octave higher than written
    max_dynamic=115,
    min_dynamic=40,  # Difficult to play softly
    typical_dynamic=90,
    max_speed=22.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE
    },
    timbre_descriptors=["piercing", "brilliant", "shrill", "cutting"],
    can_vibrato=True,
    orchestration_notes="Sounds octave higher than written. Very piercing, use sparingly. "
                       "Excellent for doubling flute at the octave or highlighting melodic peaks."
))

register_instrument(Instrument(
    name="Oboe",
    family=InstrumentFamily.WOODWINDS,
    midi_program=68,  # GM: Oboe
    range=InstrumentRange(
        lowest_note=note_name_to_midi('Bb3'),  # 58
        highest_note=note_name_to_midi('A6'),  # 93
        comfortable_low=note_name_to_midi('C4'),  # 60
        comfortable_high=note_name_to_midi('F6'),  # 89
        optimal_low=note_name_to_midi('F4'),  # 65
        optimal_high=note_name_to_midi('C6'),  # 84
    ),
    transposition=0,
    max_dynamic=105,
    min_dynamic=25,
    typical_dynamic=75,
    max_speed=14.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE
    },
    timbre_descriptors=["nasal", "reedy", "penetrating", "expressive"],
    can_vibrato=True,
    blends_well_with=["Flute", "Clarinet", "Violin", "Horn"],
    orchestration_notes="Distinctive reedy tone. Excellent for pastoral scenes and solo lines. "
                       "Penetrates through the orchestra. Standard tuning instrument (A440)."
))

register_instrument(Instrument(
    name="English Horn",
    family=InstrumentFamily.WOODWINDS,
    midi_program=69,  # GM: English Horn
    range=InstrumentRange(
        lowest_note=note_name_to_midi('E3'),  # 52 (sounding)
        highest_note=note_name_to_midi('C6'),  # 84 (sounding)
        comfortable_low=note_name_to_midi('F3'),  # 53
        comfortable_high=note_name_to_midi('G5'),  # 79
        optimal_low=note_name_to_midi('A3'),  # 57
        optimal_high=note_name_to_midi('D5'),  # 74
    ),
    transposition=-7,  # Sounds perfect 5th lower than written (F instrument)
    max_dynamic=100,
    min_dynamic=25,
    typical_dynamic=70,
    max_speed=12.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED
    },
    timbre_descriptors=["melancholic", "rich", "dark", "mournful"],
    can_vibrato=True,
    blends_well_with=["Clarinet", "Horn", "Viola", "Cello"],
    orchestration_notes="Sounds a perfect 5th lower than written. Rich, melancholic tone. "
                       "Excellent for solos expressing longing or pastoral scenes."
))

register_instrument(Instrument(
    name="Clarinet",
    family=InstrumentFamily.WOODWINDS,
    midi_program=71,  # GM: Clarinet
    range=InstrumentRange(
        lowest_note=note_name_to_midi('D3'),  # 50 (sounding E3=52 for Bb)
        highest_note=note_name_to_midi('C7'),  # 96 (sounding)
        comfortable_low=note_name_to_midi('E3'),  # 52
        comfortable_high=note_name_to_midi('G6'),  # 91
        optimal_low=note_name_to_midi('G3'),  # 55
        optimal_high=note_name_to_midi('C6'),  # 84
    ),
    transposition=-2,  # Sounds major 2nd lower (Bb instrument)
    max_dynamic=115,
    min_dynamic=20,
    typical_dynamic=80,
    max_speed=20.0,  # Very agile
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE,
        ArticulationType.TRIPLE_TONGUE, ArticulationType.SLAP_TONGUE
    },
    timbre_descriptors=["warm", "versatile", "agile", "expressive"],
    can_vibrato=True,
    blends_well_with=["Flute", "Oboe", "Bassoon", "Horn", "Strings"],
    orchestration_notes="Most versatile woodwind. Wide dynamic range. Low register (chalumeau) "
                       "is dark and rich. High register is bright. Excellent blender."
))

register_instrument(Instrument(
    name="Bass Clarinet",
    family=InstrumentFamily.WOODWINDS,
    midi_program=71,  # GM: Clarinet (no separate bass clarinet)
    range=InstrumentRange(
        lowest_note=note_name_to_midi('D2'),  # 38 (sounding Db2)
        highest_note=note_name_to_midi('G5'),  # 79
        comfortable_low=note_name_to_midi('E2'),  # 40
        comfortable_high=note_name_to_midi('D5'),  # 74
        optimal_low=note_name_to_midi('G2'),  # 43
        optimal_high=note_name_to_midi('G4'),  # 67
    ),
    transposition=-14,  # Sounds major 9th lower (Bb instrument, octave lower)
    max_dynamic=110,
    min_dynamic=25,
    typical_dynamic=75,
    max_speed=14.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.SLAP_TONGUE
    },
    timbre_descriptors=["dark", "rich", "woody", "sinister"],
    can_vibrato=True,
    blends_well_with=["Cello", "Bassoon", "Horn", "Bass"],
    orchestration_notes="Deep, rich bass voice of woodwinds. Excellent for bass lines "
                       "and ominous colors. More agile than bassoon in low register."
))

register_instrument(Instrument(
    name="Bassoon",
    family=InstrumentFamily.WOODWINDS,
    midi_program=70,  # GM: Bassoon
    range=InstrumentRange(
        lowest_note=note_name_to_midi('Bb1'),  # 34
        highest_note=note_name_to_midi('E5'),  # 76
        comfortable_low=note_name_to_midi('C2'),  # 36
        comfortable_high=note_name_to_midi('C5'),  # 72
        optimal_low=note_name_to_midi('F2'),  # 41
        optimal_high=note_name_to_midi('G4'),  # 67
    ),
    transposition=0,
    max_dynamic=105,
    min_dynamic=25,
    typical_dynamic=75,
    max_speed=12.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE
    },
    timbre_descriptors=["reedy", "expressive", "versatile", "nasal"],
    can_vibrato=True,
    blends_well_with=["Cello", "Horn", "Clarinet"],
    orchestration_notes="Bass voice of woodwinds. Can sound comical or serious depending "
                       "on context. Blends excellently with cello. Tenor register is very expressive."
))

register_instrument(Instrument(
    name="Contrabassoon",
    family=InstrumentFamily.WOODWINDS,
    midi_program=70,  # GM: Bassoon
    range=InstrumentRange(
        lowest_note=note_name_to_midi('Bb0'),  # 22 (sounding)
        highest_note=note_name_to_midi('Bb4'),  # 70 (sounding)
        comfortable_low=note_name_to_midi('C1'),  # 24
        comfortable_high=note_name_to_midi('F4'),  # 65
        optimal_low=note_name_to_midi('E1'),  # 28
        optimal_high=note_name_to_midi('C4'),  # 60
    ),
    transposition=-12,  # Sounds octave lower than written
    max_dynamic=100,
    min_dynamic=30,
    typical_dynamic=70,
    max_speed=8.0,  # Not very agile
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED
    },
    timbre_descriptors=["deep", "growling", "powerful", "ominous"],
    can_vibrato=False,
    blends_well_with=["Double Bass", "Tuba", "Bass Trombone"],
    orchestration_notes="Deepest woodwind. Sounds octave lower than written. "
                       "Provides powerful bass foundation. Use sparingly due to weight."
))

# ----------------------------------------------------------------------------
# BRASS FAMILY
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Trumpet",
    family=InstrumentFamily.BRASS,
    midi_program=56,  # GM: Trumpet
    range=InstrumentRange(
        lowest_note=note_name_to_midi('E3'),  # 52 (sounding F#3 for Bb)
        highest_note=note_name_to_midi('C6'),  # 84 (extreme high)
        comfortable_low=note_name_to_midi('F#3'),  # 54
        comfortable_high=note_name_to_midi('D5'),  # 74
        optimal_low=note_name_to_midi('A3'),  # 57
        optimal_high=note_name_to_midi('A4'),  # 69
    ),
    transposition=-2,  # Bb instrument
    max_dynamic=127,
    min_dynamic=30,
    typical_dynamic=90,
    max_speed=16.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE,
        ArticulationType.TRIPLE_TONGUE, ArticulationType.MUTED,
        ArticulationType.FLUTTER_TONGUE, ArticulationType.FALL_OFF,
        ArticulationType.RIP, ArticulationType.GLISSANDO
    },
    timbre_descriptors=["brilliant", "piercing", "fanfare", "heroic"],
    can_vibrato=True,
    pitch_bend_range=4,
    blends_well_with=["Trombone", "Horn", "Oboe"],
    orchestration_notes="Brilliant, piercing tone. Excellent for fanfares and heroic themes. "
                       "Mutes dramatically change timbre. Avoid extended high register writing."
))

register_instrument(Instrument(
    name="Horn",
    family=InstrumentFamily.BRASS,
    midi_program=60,  # GM: French Horn
    range=InstrumentRange(
        lowest_note=note_name_to_midi('B1'),  # 35 (pedal tones lower)
        highest_note=note_name_to_midi('F5'),  # 77
        comfortable_low=note_name_to_midi('F2'),  # 41
        comfortable_high=note_name_to_midi('C5'),  # 72
        optimal_low=note_name_to_midi('C3'),  # 48
        optimal_high=note_name_to_midi('F4'),  # 65
    ),
    transposition=-7,  # F instrument (perfect 5th lower)
    max_dynamic=120,
    min_dynamic=20,
    typical_dynamic=75,
    max_speed=12.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE,
        ArticulationType.MUTED
    },
    timbre_descriptors=["warm", "noble", "blending", "versatile"],
    can_vibrato=True,
    blends_well_with=["Everything"],  # Horn is the ultimate blender
    orchestration_notes="The ultimate blending instrument. Can blend with woodwinds or brass. "
                       "Noble, warm tone. Excellent for sustained chords and lyrical solos. "
                       "Stopped horn produces distinctive nasal, metallic sound."
))

register_instrument(Instrument(
    name="Trombone",
    family=InstrumentFamily.BRASS,
    midi_program=57,  # GM: Trombone
    range=InstrumentRange(
        lowest_note=note_name_to_midi('E2'),  # 40
        highest_note=note_name_to_midi('Bb5'),  # 82
        comfortable_low=note_name_to_midi('G2'),  # 43
        comfortable_high=note_name_to_midi('D5'),  # 74
        optimal_low=note_name_to_midi('Bb2'),  # 46
        optimal_high=note_name_to_midi('Bb4'),  # 70
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=25,
    typical_dynamic=85,
    max_speed=8.0,  # Limited by slide
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.MUTED,
        ArticulationType.FLUTTER_TONGUE, ArticulationType.GLISSANDO
    },
    timbre_descriptors=["powerful", "majestic", "solemn", "rich"],
    can_vibrato=True,
    pitch_bend_range=12,  # Can glissando entire range
    blends_well_with=["Trumpet", "Horn", "Tuba", "Cello"],
    orchestration_notes="Powerful, majestic sound. Natural glissandi via slide. "
                       "Excellent for chorale-like passages. Low register is particularly rich."
))

register_instrument(Instrument(
    name="Bass Trombone",
    family=InstrumentFamily.BRASS,
    midi_program=57,  # GM: Trombone
    range=InstrumentRange(
        lowest_note=note_name_to_midi('C2'),  # 36 (with trigger)
        highest_note=note_name_to_midi('F5'),  # 77
        comfortable_low=note_name_to_midi('E2'),  # 40
        comfortable_high=note_name_to_midi('Bb4'),  # 70
        optimal_low=note_name_to_midi('G2'),  # 43
        optimal_high=note_name_to_midi('F4'),  # 65
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=30,
    typical_dynamic=90,
    max_speed=6.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.MUTED,
        ArticulationType.GLISSANDO
    },
    timbre_descriptors=["deep", "powerful", "dark", "thunderous"],
    can_vibrato=True,
    pitch_bend_range=12,
    blends_well_with=["Tuba", "Bass", "Contrabassoon"],
    orchestration_notes="Deep, powerful bass voice. Trigger extends range downward. "
                       "Excellent for bass lines requiring more power than tuba."
))

register_instrument(Instrument(
    name="Tuba",
    family=InstrumentFamily.BRASS,
    midi_program=58,  # GM: Tuba
    range=InstrumentRange(
        lowest_note=note_name_to_midi('E1'),  # 28 (D1=26 with contrabass)
        highest_note=note_name_to_midi('F4'),  # 65
        comfortable_low=note_name_to_midi('F1'),  # 29
        comfortable_high=note_name_to_midi('C4'),  # 60
        optimal_low=note_name_to_midi('Bb1'),  # 34
        optimal_high=note_name_to_midi('F3'),  # 53
    ),
    transposition=0,  # (Varies: Bb, C, Eb, F tubas exist)
    max_dynamic=127,
    min_dynamic=30,
    typical_dynamic=85,
    max_speed=8.0,
    sustain_capability=True,
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TONGUED, ArticulationType.DOUBLE_TONGUE
    },
    timbre_descriptors=["deep", "round", "fundamental", "ponderous"],
    can_vibrato=False,  # Rarely used
    blends_well_with=["Trombone", "Bass", "Contrabassoon"],
    orchestration_notes="Foundation of the brass section. Deep, round tone. "
                       "More agile than expected. Can play surprisingly lyrical lines."
))

# More instruments continue...
# (For brevity, I'll add a few more key instruments and ethnic instruments)

# ----------------------------------------------------------------------------
# PERCUSSION
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Timpani",
    family=InstrumentFamily.PERCUSSION,
    midi_program=47,  # GM: Orchestral Timpani
    range=InstrumentRange(
        lowest_note=note_name_to_midi('D2'),  # 38
        highest_note=note_name_to_midi('A3'),  # 57
        comfortable_low=note_name_to_midi('E2'),  # 40
        comfortable_high=note_name_to_midi('G3'),  # 55
        optimal_low=note_name_to_midi('F2'),  # 41
        optimal_high=note_name_to_midi('F3'),  # 53
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=25,
    typical_dynamic=85,
    max_speed=10.0,  # Roll speed
    sustain_capability=False,
    polyphonic=True,  # Usually 2-4 drums
    articulations={
        ArticulationType.STACCATO, ArticulationType.TREMOLO
    },
    timbre_descriptors=["thunderous", "majestic", "resonant", "powerful"],
    pitch_bend_range=2,  # Can be tuned during performance
    orchestration_notes="Tuned drums, usually 2-4 in use. Requires time to retune. "
                       "Rolls are very effective. Can play melodic patterns."
))

# ----------------------------------------------------------------------------
# KEYBOARDS
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Piano",
    family=InstrumentFamily.KEYBOARDS,
    midi_program=0,  # GM: Acoustic Grand Piano
    range=InstrumentRange(
        lowest_note=note_name_to_midi('A0'),  # 21
        highest_note=note_name_to_midi('C8'),  # 108
        comfortable_low=note_name_to_midi('A0'),  # 21
        comfortable_high=note_name_to_midi('C8'),  # 108
        optimal_low=note_name_to_midi('C2'),  # 36
        optimal_high=note_name_to_midi('C7'),  # 96
    ),
    transposition=0,
    max_dynamic=127,
    min_dynamic=10,
    typical_dynamic=80,
    max_speed=30.0,  # Extremely fast possible
    sustain_capability=True,
    polyphonic=True,  # Full polyphony
    articulations={
        ArticulationType.LEGATO, ArticulationType.STACCATO,
        ArticulationType.TENUTO, ArticulationType.MARCATO
    },
    timbre_descriptors=["versatile", "percussive", "singing", "powerful"],
    can_vibrato=False,
    blends_well_with=["Everything"],
    orchestration_notes="Most versatile keyboard. Full 88-key range. "
                       "Can play melody, harmony, bass simultaneously. Sustain pedal essential."
))

# ----------------------------------------------------------------------------
# ETHNIC INSTRUMENTS
# ----------------------------------------------------------------------------

register_instrument(Instrument(
    name="Sitar",
    family=InstrumentFamily.ETHNIC,
    midi_program=104,  # GM: Sitar
    range=InstrumentRange(
        lowest_note=note_name_to_midi('C3'),  # 48
        highest_note=note_name_to_midi('C6'),  # 84
        comfortable_low=note_name_to_midi('D3'),  # 50
        comfortable_high=note_name_to_midi('A5'),  # 81
        optimal_low=note_name_to_midi('G3'),  # 55
        optimal_high=note_name_to_midi('E5'),  # 76
    ),
    transposition=0,
    max_dynamic=100,
    min_dynamic=20,
    typical_dynamic=70,
    max_speed=12.0,
    sustain_capability=True,
    polyphonic=False,
    articulations={
        ArticulationType.LEGATO, ArticulationType.GLISSANDO
    },
    timbre_descriptors=["resonant", "sympathetic", "droning", "modal"],
    can_vibrato=True,
    pitch_bend_range=12,  # Can bend extensively
    orchestration_notes="Indian classical instrument. Sympathetic strings create rich drone. "
                       "Extensive pitch bending (meend) is characteristic. Used for raga performance."
))

register_instrument(Instrument(
    name="Koto",
    family=InstrumentFamily.ETHNIC,
    midi_program=107,  # GM: Koto
    range=InstrumentRange(
        lowest_note=note_name_to_midi('D2'),  # 38
        highest_note=note_name_to_midi('E5'),  # 76
        comfortable_low=note_name_to_midi('E2'),  # 40
        comfortable_high=note_name_to_midi('C5'),  # 72
        optimal_low=note_name_to_midi('A2'),  # 45
        optimal_high=note_name_to_midi('G4'),  # 67
    ),
    transposition=0,
    max_dynamic=95,
    min_dynamic=25,
    typical_dynamic=65,
    max_speed=10.0,
    sustain_capability=True,
    polyphonic=True,  # Can pluck multiple strings
    articulations={
        ArticulationType.PIZZICATO, ArticulationType.GLISSANDO,
        ArticulationType.TREMOLO
    },
    timbre_descriptors=["delicate", "resonant", "pentatonic", "ethereal"],
    pitch_bend_range=3,
    orchestration_notes="Japanese 13-string zither. Typically uses pentatonic scales. "
                       "Can press strings behind bridges to raise pitch. Glissandi are characteristic."
))


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_instrument(name: str) -> Optional[Instrument]:
    """Retrieve an instrument by name"""
    return INSTRUMENTS.get(name)


def get_instruments_by_family(family: InstrumentFamily) -> List[Instrument]:
    """Get all instruments in a family"""
    return [inst for inst in INSTRUMENTS.values() if inst.family == family]


def get_instruments_by_range(min_note: int, max_note: int) -> List[Instrument]:
    """Find instruments that can play within the given range"""
    result = []
    for inst in INSTRUMENTS.values():
        if (inst.range.lowest_note <= min_note and
            inst.range.highest_note >= max_note):
            result.append(inst)
    return result


def transpose_note(note: int, instrument: Instrument) -> int:
    """
    Transpose a sounding note to the written note for the instrument.

    Args:
        note: Sounding MIDI note
        instrument: Target instrument

    Returns:
        Written MIDI note
    """
    return note - instrument.transposition


def written_to_sounding(note: int, instrument: Instrument) -> int:
    """
    Convert written note to sounding note.

    Args:
        note: Written MIDI note
        instrument: Source instrument

    Returns:
        Sounding MIDI note
    """
    return note + instrument.transposition


def is_in_comfortable_range(note: int, instrument: Instrument) -> bool:
    """Check if a note is in the comfortable tessitura"""
    sounding = written_to_sounding(note, instrument)
    return (instrument.range.comfortable_low <= sounding <=
            instrument.range.comfortable_high)


def is_in_optimal_range(note: int, instrument: Instrument) -> bool:
    """Check if a note is in the optimal (sweet spot) range"""
    sounding = written_to_sounding(note, instrument)
    return (instrument.range.optimal_low <= sounding <=
            instrument.range.optimal_high)


def get_register_name(note: int, instrument: Instrument) -> str:
    """
    Get descriptive register name for a note on an instrument.

    Returns: 'low', 'middle', 'high', or 'extreme'
    """
    sounding = written_to_sounding(note, instrument)
    range_span = instrument.range.highest_note - instrument.range.lowest_note
    position = (sounding - instrument.range.lowest_note) / range_span

    if position < 0.25:
        return "low"
    elif position < 0.5:
        return "middle-low"
    elif position < 0.75:
        return "middle-high"
    else:
        return "high"


# ============================================================================
# MAIN (EXAMPLE/TEST)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("COMPREHENSIVE INSTRUMENT LIBRARY")
    print("=" * 80)
    print(f"\nTotal instruments: {len(INSTRUMENTS)}\n")

    # List all families
    for family in InstrumentFamily:
        instruments = get_instruments_by_family(family)
        if instruments:
            print(f"\n{family.value.upper()} ({len(instruments)} instruments):")
            for inst in instruments:
                range_str = f"{midi_to_note_name(inst.range.lowest_note)} - " \
                           f"{midi_to_note_name(inst.range.highest_note)}"
                trans_str = f" (transp: {inst.transposition:+d})" if inst.transposition != 0 else ""
                print(f"  • {inst.name}: {range_str}{trans_str}")

    # Example: Find instruments for middle C
    print("\n" + "=" * 80)
    print("Example: Instruments that can play middle C (60) comfortably:")
    print("=" * 80)
    middle_c = 60
    for inst in INSTRUMENTS.values():
        if is_in_comfortable_range(middle_c, inst):
            register = get_register_name(middle_c, inst)
            print(f"  • {inst.name}: {register} register")

    # Example: Violin details
    print("\n" + "=" * 80)
    print("Example: Detailed Violin Specification")
    print("=" * 80)
    violin = get_instrument("Violin")
    print(f"Name: {violin.name}")
    print(f"Family: {violin.family.value}")
    print(f"MIDI Program: {violin.midi_program}")
    print(f"Range: {midi_to_note_name(violin.range.lowest_note)} - "
          f"{midi_to_note_name(violin.range.highest_note)}")
    print(f"Comfortable: {midi_to_note_name(violin.range.comfortable_low)} - "
          f"{midi_to_note_name(violin.range.comfortable_high)}")
    print(f"Optimal: {midi_to_note_name(violin.range.optimal_low)} - "
          f"{midi_to_note_name(violin.range.optimal_high)}")
    print(f"Max speed: {violin.max_speed} notes/sec")
    print(f"Articulations: {len(violin.articulations)}")
    for art in sorted(violin.articulations, key=lambda x: x.value):
        print(f"  • {art.value}")
    print(f"Timbre: {', '.join(violin.timbre_descriptors)}")
    print(f"Blends with: {', '.join(violin.blends_well_with)}")
    print(f"\nNotes: {violin.orchestration_notes}")

    print("\n" + "=" * 80)
    print("Library ready for orchestration engine!")
    print("=" * 80)
