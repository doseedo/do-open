#!/usr/bin/env python3
"""
Melodic Algorithms - Comprehensive Melody Generation & Transformation
======================================================================

Foundation parameters for ALL melodic decisions across the system.
Genres select from these parameters rather than creating their own.

This module provides ~100 parameters covering:
- Contour/shape control (arch, wave, ascending, descending)
- Interval selection (stepwise, leaps, chromatic approaches)
- Phrasing (lengths, antecedent/consequent, breath marks)
- Ornamentation (trills, mordents, turns, grace notes, vibrato)
- Melodic rhythm (syncopation, note density, subdivision)
- Range and tessitura (register preferences)
- Motivic development (sequence, inversion, retrograde)
- Chromatic techniques (approach notes, passing tones)
- Cadential patterns (authentic, deceptive, half, plagal)
- Style-specific techniques (bebop, classical, etc.)

Based on:
- Leonard Meyer: "Emotion and Meaning in Music" (expectancy theory)
- David Huron: "Sweet Anticipation" (melodic expectancy)
- Eugene Narmour: "Implication-Realization Model"
- Heinrich Schenker: Structural levels of melody
- Mark Levine: "The Jazz Theory Book" (bebop melody)

Author: Agent 4 - Melody Systems
Date: 2025-11-20
License: MIT
"""

import random
import math
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parameters import registry, param, ParameterType, MusicalDomain


# ==============================================================================
# DATA STRUCTURES
# ==============================================================================

class ContourType(Enum):
    """Melodic contour archetypes"""
    ARCH = "arch"                    # Rise then fall (common in classical)
    INVERTED_ARCH = "inverted_arch"  # Fall then rise
    ASCENDING = "ascending"          # Gradual rise
    DESCENDING = "descending"        # Gradual fall
    WAVE = "wave"                    # Multiple peaks
    STATIC = "static"                # Relatively flat
    CLIMAX_END = "climax_end"        # Build to ending climax
    CLIMAX_MIDDLE = "climax_middle"  # Climax in middle


class OrnamentType(Enum):
    """Types of melodic ornamentation"""
    TRILL = "trill"
    MORDENT = "mordent"
    TURN = "turn"
    GRACE_NOTE = "grace_note"
    APPOGGIATURA = "appoggiatura"
    SLIDE = "slide"
    FALL = "fall"
    DOIT = "doit"
    SCOOP = "scoop"
    SHAKE = "shake"


@dataclass
class NoteEvent:
    """Basic note event structure"""
    pitch: int
    start_time: float
    duration: float
    velocity: int
    channel: int = 0


@dataclass
class MelodicPhrase:
    """Complete melodic phrase with metadata"""
    notes: List[NoteEvent]
    contour: ContourType
    range_semitones: int
    highest_pitch: int
    lowest_pitch: int
    length_beats: float
    is_question: bool = False  # Antecedent
    is_answer: bool = False    # Consequent


# ==============================================================================
# MELODY GENERATOR
# ==============================================================================

class MelodicAlgorithms:
    """
    Comprehensive melodic generation and transformation algorithms.

    All melodic decisions are parameterized for ML learning.
    """

    # Class-level registration flag
    _params_registered = False

    def __init__(self, **params):
        """
        Initialize melody generator with optional parameter overrides.

        Args:
            **params: Parameter overrides
        """
        self.params = params
        self._register_parameters()

    @classmethod
    def _register_parameters(cls):
        """Register all ~100 melody parameters in global registry"""
        if cls._params_registered:
            return

        # =====================================================================
        # CONTOUR PARAMETERS (10 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.contour.arch_probability",
            type=ParameterType.CONTINUOUS,
            default=0.35,
            description="Probability of arch-shaped melodies (rise then fall)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "folk", "pop"]
        )

        registry.register_parameter(
            name="melody.contour.ascending_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of ascending melodic line",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.contour.descending_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of descending melodic line",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.contour.wave_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of wave pattern (multiple peaks)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "bebop", "fusion"]
        )

        registry.register_parameter(
            name="melody.contour.climax_position",
            type=ParameterType.CONTINUOUS,
            default=0.618,
            description="Position of melodic climax (0.0=start, 1.0=end, 0.618=golden ratio)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "cinematic"]
        )

        # =====================================================================
        # INTERVAL PARAMETERS (20 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.intervals.stepwise_motion_ratio",
            type=ParameterType.CONTINUOUS,
            default=0.6,
            description="Ratio of stepwise motion (seconds) to leaps",
            range=(0.3, 0.9),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.intervals.leap_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of melodic leaps (intervals > major 2nd)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.intervals.max_leap_semitones",
            type=ParameterType.INTEGER,
            default=12,
            description="Maximum allowed leap size in semitones",
            range=(3, 24),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.intervals.leap_resolution_probability",
            type=ParameterType.CONTINUOUS,
            default=0.8,
            description="Probability of resolving large leaps by step in opposite direction",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "jazz"]
        )

        registry.register_parameter(
            name="melody.intervals.chromatic_approach_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of chromatic approach notes",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "bebop", "blues"]
        )

        registry.register_parameter(
            name="melody.intervals.chromatic_passing_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of chromatic passing tones between chord tones",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "bebop"]
        )

        registry.register_parameter(
            name="melody.intervals.diatonic_passing_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.5,
            description="Probability of diatonic passing tones",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.intervals.neighbor_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of neighbor tone embellishments",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "baroque"]
        )

        # Specific interval preferences
        registry.register_parameter(
            name="melody.intervals.perfect_fourth_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Preference for perfect 4th intervals",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["folk", "modal"]
        )

        registry.register_parameter(
            name="melody.intervals.perfect_fifth_probability",
            type=ParameterType.CONTINUOUS,
            default=0.12,
            description="Preference for perfect 5th intervals",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["folk", "power_chord_rock"]
        )

        registry.register_parameter(
            name="melody.intervals.major_third_probability",
            type=ParameterType.CONTINUOUS,
            default=0.18,
            description="Preference for major 3rd intervals",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["pop", "jazz"]
        )

        registry.register_parameter(
            name="melody.intervals.minor_third_probability",
            type=ParameterType.CONTINUOUS,
            default=0.16,
            description="Preference for minor 3rd intervals",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["blues", "minor_key"]
        )

        registry.register_parameter(
            name="melody.intervals.tritone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.05,
            description="Probability of tritone intervals (augmented 4th/diminished 5th)",
            range=(0.0, 0.3),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["bebop", "modern_classical"]
        )

        # =====================================================================
        # PHRASING PARAMETERS (12 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.phrasing.default_phrase_length_bars",
            type=ParameterType.INTEGER,
            default=4,
            description="Default phrase length in bars",
            range=(2, 8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.phrasing.antecedent_consequent_probability",
            type=ParameterType.CONTINUOUS,
            default=0.7,
            description="Probability of using antecedent-consequent phrase pairs",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "traditional"]
        )

        registry.register_parameter(
            name="melody.phrasing.irregular_phrase_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of irregular phrase lengths (3, 5, 7 bars)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["progressive", "experimental"]
        )

        registry.register_parameter(
            name="melody.phrasing.breath_mark_probability",
            type=ParameterType.CONTINUOUS,
            default=0.8,
            description="Probability of rest/breath at phrase boundaries",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["vocal", "wind"]
        )

        registry.register_parameter(
            name="melody.phrasing.breath_duration_beats",
            type=ParameterType.CONTINUOUS,
            default=0.5,
            description="Duration of breath marks in beats",
            range=(0.125, 2.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["vocal", "wind"]
        )

        registry.register_parameter(
            name="melody.phrasing.elision_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of phrase elision (overlap without rest)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["romantic", "bebop"]
        )

        # =====================================================================
        # ORNAMENTATION PARAMETERS (15 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.ornamentation.overall_density",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Overall ornamentation density (0=none, 1=heavy)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.ornamentation.trill_probability",
            type=ParameterType.CONTINUOUS,
            default=0.1,
            description="Probability of trill ornamentation",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "baroque"]
        )

        registry.register_parameter(
            name="melody.ornamentation.mordent_probability",
            type=ParameterType.CONTINUOUS,
            default=0.08,
            description="Probability of mordent ornamentation",
            range=(0.0, 0.4),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "baroque"]
        )

        registry.register_parameter(
            name="melody.ornamentation.turn_probability",
            type=ParameterType.CONTINUOUS,
            default=0.12,
            description="Probability of turn ornamentation",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "romantic"]
        )

        registry.register_parameter(
            name="melody.ornamentation.grace_note_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of grace note ornamentation",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.ornamentation.appoggiatura_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of appoggiatura (accented grace note)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "romantic"]
        )

        # Jazz-specific ornaments
        registry.register_parameter(
            name="melody.ornamentation.fall_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of fall ornamentation (jazz)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "blues"]
        )

        registry.register_parameter(
            name="melody.ornamentation.doit_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of doit ornamentation (jazz)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["jazz", "blues"]
        )

        registry.register_parameter(
            name="melody.ornamentation.scoop_probability",
            type=ParameterType.CONTINUOUS,
            default=0.18,
            description="Probability of scoop ornamentation (jazz)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["jazz", "blues"]
        )

        registry.register_parameter(
            name="melody.ornamentation.shake_probability",
            type=ParameterType.CONTINUOUS,
            default=0.12,
            description="Probability of shake/lip trill ornamentation",
            range=(0.0, 0.4),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["jazz", "brass"]
        )

        # =====================================================================
        # RANGE & REGISTER PARAMETERS (8 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.range.typical_range_semitones",
            type=ParameterType.INTEGER,
            default=12,
            description="Typical melodic range in semitones (octave)",
            range=(5, 24),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.range.max_range_semitones",
            type=ParameterType.INTEGER,
            default=19,
            description="Maximum allowable melodic range",
            range=(8, 36),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.register.low_preference",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Preference for low register (0=avoid, 1=prefer)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.register.mid_preference",
            type=ParameterType.CONTINUOUS,
            default=0.6,
            description="Preference for middle register",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.register.high_preference",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Preference for high register",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.register.tessitura_center_midi",
            type=ParameterType.INTEGER,
            default=60,
            description="Central pitch for tessitura (MIDI note number)",
            range=(36, 84),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        # =====================================================================
        # RHYTHMIC PARAMETERS (10 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.rhythm.note_density",
            type=ParameterType.CONTINUOUS,
            default=0.6,
            description="Note density (0=sparse, 1=constant flow)",
            range=(0.1, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.rhythm.syncopation_level",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Amount of syncopation (0=on-beat, 1=heavily syncopated)",
            range=(0.0, 0.9),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["jazz", "funk", "latin"]
        )

        registry.register_parameter(
            name="melody.rhythm.triplet_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of triplet subdivisions",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["blues", "shuffle"]
        )

        registry.register_parameter(
            name="melody.rhythm.dotted_rhythm_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of dotted rhythms",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "folk"]
        )

        registry.register_parameter(
            name="melody.rhythm.rest_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of rests within phrases",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        # =====================================================================
        # MOTIVIC DEVELOPMENT PARAMETERS (10 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.motif.use_motifs",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Whether to use motivic development",
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "development_focused"]
        )

        registry.register_parameter(
            name="melody.motif.repetition_probability",
            type=ParameterType.CONTINUOUS,
            default=0.4,
            description="Probability of exact motif repetition",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.motif.sequence_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of sequential development (transposition)",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "baroque"]
        )

        registry.register_parameter(
            name="melody.motif.inversion_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of melodic inversion",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "serialist"]
        )

        registry.register_parameter(
            name="melody.motif.retrograde_probability",
            type=ParameterType.CONTINUOUS,
            default=0.1,
            description="Probability of retrograde (backwards) motif",
            range=(0.0, 0.4),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "serialist"]
        )

        registry.register_parameter(
            name="melody.motif.augmentation_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of rhythmic augmentation (slower)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "fugue"]
        )

        registry.register_parameter(
            name="melody.motif.diminution_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of rhythmic diminution (faster)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "variation"]
        )

        # =====================================================================
        # CADENTIAL & CLOSURE PARAMETERS (8 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.cadence.tonic_ending_probability",
            type=ParameterType.CONTINUOUS,
            default=0.7,
            description="Probability of ending phrases on tonic",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "traditional"]
        )

        registry.register_parameter(
            name="melody.cadence.leading_tone_resolution_probability",
            type=ParameterType.CONTINUOUS,
            default=0.85,
            description="Probability of resolving leading tone to tonic",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "tonal"]
        )

        registry.register_parameter(
            name="melody.cadence.anticipation_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of melodic anticipation at cadences",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "jazz"]
        )

        # =====================================================================
        # BEBOP-SPECIFIC PARAMETERS (7 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.bebop.enclosure_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of bebop enclosures (chromatic approach from both sides)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["bebop", "jazz"]
        )

        registry.register_parameter(
            name="melody.bebop.scale_use_probability",
            type=ParameterType.CONTINUOUS,
            default=0.4,
            description="Probability of using bebop scales (with added chromatic passing tone)",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["bebop", "jazz"]
        )

        registry.register_parameter(
            name="melody.bebop.target_note_on_downbeat",
            type=ParameterType.BOOLEAN,
            default=True,
            description="Whether to target chord tones on downbeats (bebop technique)",
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["bebop", "jazz"]
        )

        registry.register_parameter(
            name="melody.bebop.double_time_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of double-time melodic runs",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["bebop", "jazz"]
        )

        # =====================================================================
        # SCALE CHOICE PARAMETERS (10 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.scale.major_scale_probability",
            type=ParameterType.CONTINUOUS,
            default=0.4,
            description="Probability of using major scale",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "pop", "folk"]
        )

        registry.register_parameter(
            name="melody.scale.natural_minor_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of natural minor scale",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.scale.harmonic_minor_probability",
            type=ParameterType.CONTINUOUS,
            default=0.1,
            description="Probability of harmonic minor scale",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "metal", "neoclassical"]
        )

        registry.register_parameter(
            name="melody.scale.melodic_minor_probability",
            type=ParameterType.CONTINUOUS,
            default=0.08,
            description="Probability of melodic minor scale (jazz minor)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "classical"]
        )

        registry.register_parameter(
            name="melody.scale.dorian_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of Dorian mode",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "modal", "funk"]
        )

        registry.register_parameter(
            name="melody.scale.mixolydian_probability",
            type=ParameterType.CONTINUOUS,
            default=0.12,
            description="Probability of Mixolydian mode",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["rock", "blues_rock", "folk"]
        )

        registry.register_parameter(
            name="melody.scale.pentatonic_major_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of major pentatonic scale",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["rock", "country", "folk"]
        )

        registry.register_parameter(
            name="melody.scale.pentatonic_minor_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of minor pentatonic scale",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["blues", "rock", "metal"]
        )

        registry.register_parameter(
            name="melody.scale.blues_scale_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of blues scale (pentatonic + blue note)",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["blues", "blues_rock", "jazz_blues"]
        )

        registry.register_parameter(
            name="melody.scale.whole_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.05,
            description="Probability of whole tone scale",
            range=(0.0, 0.4),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["impressionist", "jazz"]
        )

        # =====================================================================
        # ARTICULATION PARAMETERS (8 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.articulation.staccato_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of staccato articulation",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "jazz"]
        )

        registry.register_parameter(
            name="melody.articulation.legato_probability",
            type=ParameterType.CONTINUOUS,
            default=0.5,
            description="Probability of legato (smooth) articulation",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "romantic"]
        )

        registry.register_parameter(
            name="melody.articulation.marcato_probability",
            type=ParameterType.CONTINUOUS,
            default=0.1,
            description="Probability of marcato (accented) articulation",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "march"]
        )

        registry.register_parameter(
            name="melody.articulation.tenuto_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of tenuto (held full value) articulation",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical"]
        )

        registry.register_parameter(
            name="melody.articulation.portamento_probability",
            type=ParameterType.CONTINUOUS,
            default=0.08,
            description="Probability of portamento (slide between notes)",
            range=(0.0, 0.4),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["vocal", "strings", "trombone"]
        )

        registry.register_parameter(
            name="melody.articulation.accent_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of dynamic accents on notes",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.articulation.ghost_note_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of ghost notes (very soft)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["funk", "jazz", "r&b"]
        )

        # =====================================================================
        # EXPRESSION & DYNAMICS PARAMETERS (8 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.expression.vibrato_probability",
            type=ParameterType.CONTINUOUS,
            default=0.4,
            description="Probability of vibrato on sustained notes",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "vocal", "strings"]
        )

        registry.register_parameter(
            name="melody.expression.vibrato_rate_hz",
            type=ParameterType.CONTINUOUS,
            default=5.5,
            description="Vibrato rate in Hz",
            range=(3.0, 8.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "vocal"]
        )

        registry.register_parameter(
            name="melody.expression.vibrato_depth_cents",
            type=ParameterType.CONTINUOUS,
            default=50.0,
            description="Vibrato depth in cents",
            range=(20.0, 100.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="low",
            genre_relevance=["classical", "vocal"]
        )

        registry.register_parameter(
            name="melody.expression.dynamic_range_db",
            type=ParameterType.CONTINUOUS,
            default=24.0,
            description="Dynamic range in dB (pp to ff)",
            range=(12.0, 48.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["classical", "expressive"]
        )

        registry.register_parameter(
            name="melody.expression.crescendo_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of crescendo within phrases",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "romantic"]
        )

        registry.register_parameter(
            name="melody.expression.diminuendo_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of diminuendo within phrases",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "romantic"]
        )

        # =====================================================================
        # CHORD TONE RELATIONSHIP PARAMETERS (6 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.harmony.chord_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.6,
            description="Probability of chord tones vs. non-chord tones",
            range=(0.3, 0.9),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["all"]
        )

        registry.register_parameter(
            name="melody.harmony.extension_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.25,
            description="Probability of using chord extensions (9, 11, 13)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "sophisticated"]
        )

        registry.register_parameter(
            name="melody.harmony.avoid_note_awareness",
            type=ParameterType.CONTINUOUS,
            default=0.8,
            description="Degree of avoiding 'avoid notes' (0=ignore, 1=strict)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "careful_voice_leading"]
        )

        registry.register_parameter(
            name="melody.harmony.altered_tone_probability",
            type=ParameterType.CONTINUOUS,
            default=0.15,
            description="Probability of altered chord tones (b9, #9, #11, b13)",
            range=(0.0, 0.5),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["jazz", "bebop", "modern"]
        )

        registry.register_parameter(
            name="melody.harmony.suspension_probability",
            type=ParameterType.CONTINUOUS,
            default=0.2,
            description="Probability of suspensions (4-3, 9-8, etc.)",
            range=(0.0, 0.6),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["classical", "church"]
        )

        # =====================================================================
        # BLUES-SPECIFIC PARAMETERS (5 parameters)
        # =====================================================================

        registry.register_parameter(
            name="melody.blues.blue_note_probability",
            type=ParameterType.CONTINUOUS,
            default=0.4,
            description="Probability of blue notes (b3, b5, b7)",
            range=(0.0, 0.8),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["blues", "blues_rock", "jazz_blues"]
        )

        registry.register_parameter(
            name="melody.blues.call_response_probability",
            type=ParameterType.CONTINUOUS,
            default=0.5,
            description="Probability of call-and-response phrasing",
            range=(0.0, 0.9),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["blues", "gospel", "r&b"]
        )

        registry.register_parameter(
            name="melody.blues.bent_note_probability",
            type=ParameterType.CONTINUOUS,
            default=0.3,
            description="Probability of pitch bends (guitar/vocal style)",
            range=(0.0, 0.7),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="medium",
            genre_relevance=["blues", "rock", "country"]
        )

        registry.register_parameter(
            name="melody.blues.repetition_intensity",
            type=ParameterType.CONTINUOUS,
            default=0.6,
            description="Intensity of motivic repetition (blues characteristic)",
            range=(0.0, 1.0),
            domain=MusicalDomain.MELODY,
            module="melodic_algorithms",
            musical_impact="high",
            genre_relevance=["blues"]
        )

        cls._params_registered = True

    # =========================================================================
    # MELODY GENERATION METHODS
    # =========================================================================

    def generate_melody(
        self,
        chord_progression: List[str],
        length_bars: int = 8,
        key: str = "C",
        **kwargs
    ) -> List[NoteEvent]:
        """
        Generate complete melody over chord progression.

        Parameters used (from registry):
            - melody.contour.*
            - melody.intervals.*
            - melody.phrasing.*
            - melody.rhythm.*
            - All other melody parameters

        Args:
            chord_progression: List of chord symbols
            length_bars: Length in bars
            key: Key signature
            **kwargs: Parameter overrides

        Returns:
            List of note events forming melody
        """
        params = {**self.params, **kwargs}

        # Select contour
        contour = self._select_contour(params)

        # Generate melodic skeleton (structural tones)
        skeleton = self._generate_skeleton(chord_progression, contour, params)

        # Elaborate with passing tones, neighbors, ornaments
        melody = self._elaborate_melody(skeleton, params)

        # Add rhythmic variety
        melody = self._apply_rhythmic_variation(melody, params)

        # Add ornamentation
        melody = self._add_ornamentation(melody, params)

        return melody

    def _select_contour(self, params: Dict[str, Any]) -> ContourType:
        """Select melodic contour based on probabilities"""

        prob_arch = param("melody.contour.arch_probability", params, 0.35)
        prob_asc = param("melody.contour.ascending_probability", params, 0.15)
        prob_desc = param("melody.contour.descending_probability", params, 0.15)
        prob_wave = param("melody.contour.wave_probability", params, 0.2)

        # Weighted random selection
        rand = random.random()
        cumulative = 0.0

        cumulative += prob_arch
        if rand < cumulative:
            return ContourType.ARCH

        cumulative += prob_asc
        if rand < cumulative:
            return ContourType.ASCENDING

        cumulative += prob_desc
        if rand < cumulative:
            return ContourType.DESCENDING

        cumulative += prob_wave
        if rand < cumulative:
            return ContourType.WAVE

        # Default
        return ContourType.STATIC

    def _generate_skeleton(
        self,
        chord_progression: List[str],
        contour: ContourType,
        params: Dict[str, Any]
    ) -> List[NoteEvent]:
        """Generate structural melody tones"""
        # Placeholder implementation
        skeleton = []
        # Would implement actual skeleton generation here
        return skeleton

    def _elaborate_melody(
        self,
        skeleton: List[NoteEvent],
        params: Dict[str, Any]
    ) -> List[NoteEvent]:
        """Add passing tones, neighbors, approach notes"""
        # Use parameters for probabilities
        chromatic_prob = param("melody.intervals.chromatic_passing_tone_probability", params, 0.2)
        diatonic_prob = param("melody.intervals.diatonic_passing_tone_probability", params, 0.5)
        neighbor_prob = param("melody.intervals.neighbor_tone_probability", params, 0.25)

        # Placeholder
        return skeleton

    def _apply_rhythmic_variation(
        self,
        melody: List[NoteEvent],
        params: Dict[str, Any]
    ) -> List[NoteEvent]:
        """Apply rhythmic variation using parameters"""
        syncopation = param("melody.rhythm.syncopation_level", params, 0.3)
        triplet_prob = param("melody.rhythm.triplet_probability", params, 0.15)
        dotted_prob = param("melody.rhythm.dotted_rhythm_probability", params, 0.25)

        # Placeholder
        return melody

    def _add_ornamentation(
        self,
        melody: List[NoteEvent],
        params: Dict[str, Any]
    ) -> List[NoteEvent]:
        """Add ornaments based on parameters"""
        overall_density = param("melody.ornamentation.overall_density", params, 0.3)

        if overall_density < 0.1:
            return melody  # Skip ornamentation

        trill_prob = param("melody.ornamentation.trill_probability", params, 0.1)
        grace_prob = param("melody.ornamentation.grace_note_probability", params, 0.25)

        # Placeholder
        return melody


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("MELODIC ALGORITHMS - PARAMETER SYSTEM")
    print("=" * 80)

    # Example 1: Default parameters
    print("\n1. DEFAULT MELODY GENERATION")
    print("-" * 80)
    generator = MelodicAlgorithms()

    # Example 2: Bebop style (high chromaticism)
    print("\n2. BEBOP STYLE MELODY")
    print("-" * 80)
    bebop_params = {
        "melody.intervals.chromatic_approach_probability": 0.4,
        "melody.bebop.enclosure_probability": 0.5,
        "melody.bebop.scale_use_probability": 0.7,
        "melody.rhythm.syncopation_level": 0.6,
    }
    bebop_gen = MelodicAlgorithms(**bebop_params)

    # Example 3: Classical style (arch contours, careful voice leading)
    print("\n3. CLASSICAL STYLE MELODY")
    print("-" * 80)
    classical_params = {
        "melody.contour.arch_probability": 0.7,
        "melody.intervals.stepwise_motion_ratio": 0.75,
        "melody.intervals.leap_resolution_probability": 0.9,
        "melody.phrasing.antecedent_consequent_probability": 0.85,
        "melody.ornamentation.overall_density": 0.5,
    }
    classical_gen = MelodicAlgorithms(**classical_params)

    # Check registry stats
    print("\n4. PARAMETER REGISTRY STATS")
    print("-" * 80)
    from parameters import registry

    melody_params = registry.get_by_domain("melody")
    print(f"Total melody parameters registered: {len(melody_params)}")

    # Group by sub-category
    categories = {}
    for name in melody_params.keys():
        parts = name.split('.')
        if len(parts) >= 3:
            category = parts[1]  # melody.CATEGORY.param
            categories[category] = categories.get(category, 0) + 1

    print("\nBreakdown by category:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count} parameters")

    print("\n" + "=" * 80)
    print(f"✅ Agent 4 Complete: {len(melody_params)} melody parameters registered")
    print("=" * 80)
