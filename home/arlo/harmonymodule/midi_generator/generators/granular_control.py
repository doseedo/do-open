#!/usr/bin/env python3
"""
Granular Control System - Precision Musical Component Generation
==================================================================

This module enables Photoshop-level control over music generation by combining:
1. User-defined rhythm patterns
2. Chord progressions
3. Instrument sections (brass, strings, woodwinds, percussion)
4. Idiomatic writing rules for each instrument family

The system generates professional, idiomatic musical notation that respects
instrument ranges, articulation patterns, voicing rules, and technical limitations.

Key Features:
- Rhythm-to-notes mapping with chord awareness
- Section-specific articulation patterns (brass hits, string staccato, etc.)
- Automatic voicing for different ensembles
- Range validation and transposition handling
- Dynamic and expression mapping
- Style-appropriate ornaments and techniques

Research Sources:
-----------------
- Samuel Adler: "The Study of Orchestration" (4th Edition) - Idiomatic writing
- Rimsky-Korsakov: "Principles of Orchestration" (1913) - Instrumental techniques
- Alfred Blatter: "Instrumentation and Orchestration" (2nd ed.) - Practical ranges
- Brass articulation research (2024) - Tonguing patterns, double/triple tonguing
- String bowing techniques - Détaché, martelé, spiccato, pizzicato
- Woodwind voicing - Traditional vs. interlocking voicings
- Film scoring techniques - Hans Zimmer, John Williams brass writing
- Jazz arranging - Sammy Nestico, Thad Jones big band techniques

Author: Agent 8 - Modular Genre Fusion Enhancement
Date: 2025-11-19
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from fractions import Fraction
import random
import copy

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from existing modules
try:
    from advanced_modules.orchestration_advanced import (
        INSTRUMENT_DATABASE, InstrumentRange, Register, Playability,
        StringTechnique, DoublingStrategy, VoicingResult
    )
    from advanced_modules.chord_voicing import (
        ChordVoicing, VoicingType, ChordQuality, ChordSymbol,
        EnsembleType, parse_chord_symbol
    )
    from midi_generator.algorithms.advanced_rhythm import (
        RhythmicEvent, OddMeterStyle
    )
except ImportError as e:
    print(f"Warning: Some imports failed: {e}")
    # Create minimal fallbacks for development
    class Register(Enum):
        DARK = "dark"
        NEUTRAL = "neutral"
        BRIGHT = "bright"

    class Playability(Enum):
        EXCELLENT = 5
        GOOD = 4
        ACCEPTABLE = 3
        DIFFICULT = 2
        PROBLEMATIC = 1
        UNPLAYABLE = 0

    class StringTechnique(Enum):
        ARCO = "arco"
        PIZZICATO = "pizzicato"
        TREMOLO = "tremolo"
        STACCATO = "staccato"
        LEGATO = "legato"


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class InstrumentSection(Enum):
    """Instrument section types"""
    BRASS = "brass"
    STRINGS = "strings"
    WOODWINDS = "woodwinds"
    PERCUSSION = "percussion"
    RHYTHM_SECTION = "rhythm_section"  # Jazz/pop: piano, bass, guitar, drums
    CHOIR = "choir"  # SATB vocals


class ArticulationType(Enum):
    """Musical articulation types"""
    # Universal
    STACCATO = "staccato"          # Short, detached
    LEGATO = "legato"              # Smooth, connected
    ACCENT = "accent"              # Emphasized
    MARCATO = "marcato"            # Heavy accent
    TENUTO = "tenuto"              # Full value, slight emphasis

    # Brass-specific
    TONGUED = "tongued"            # Single tongue attack
    DOUBLE_TONGUE = "double_tongue"  # Fast repeated notes
    TRIPLE_TONGUE = "triple_tongue"  # Triplet patterns
    SLURRED = "slurred"            # Smooth connection
    FALL_OFF = "fall_off"          # Jazz fall
    DOIT = "doit"                  # Jazz scoop up
    SHAKE = "shake"                # Lip trill

    # String-specific
    DETACHE = "detache"            # Separate bows
    MARTELE = "martele"            # Hammered
    SPICCATO = "spiccato"          # Bouncing bow
    PIZZICATO = "pizzicato"        # Plucked
    COL_LEGNO = "col_legno"        # With wood of bow
    SUL_PONTICELLO = "sul_ponticello"  # Near bridge
    TREMOLO = "tremolo"            # Rapid bow movement

    # Woodwind-specific
    FLUTTER_TONGUE = "flutter_tongue"  # Flutter effect
    SLAP_TONGUE = "slap_tongue"    # Percussive attack


class VoicingStrategy(Enum):
    """Chord voicing strategies for sections"""
    CLOSE = "close"                # Notes within octave
    OPEN = "open"                  # Spread voicing
    DROP_2 = "drop_2"              # Drop second voice
    DROP_3 = "drop_3"              # Drop third voice
    TRADITIONAL = "traditional"    # Standard orchestral (fl, ob, cl, bn)
    INTERLOCKING = "interlocking"  # Mixed timbres per voice
    UNISON = "unison"              # All same pitch
    OCTAVES = "octaves"            # Octave doubling
    FOURTHS = "fourths"            # Quartal voicing
    CLUSTERS = "clusters"          # Tone clusters


class RhythmicStyle(Enum):
    """Rhythmic style for pattern interpretation"""
    STRAIGHT = "straight"          # No swing
    SWING = "swing"                # Jazz swing (triplet feel)
    SHUFFLE = "shuffle"            # Heavy swing
    HALF_TIME = "half_time"        # Half-time feel
    DOUBLE_TIME = "double_time"    # Double-time feel
    SYNCOPATED = "syncopated"      # Off-beat emphasis
    ON_BEAT = "on_beat"            # Downbeat emphasis


@dataclass
class RhythmPattern:
    """
    User-defined rhythm pattern

    Examples:
        # Brass hits on 1 and 3:
        RhythmPattern(
            onsets=[0.0, 2.0],      # Beats 1 and 3
            durations=[0.25, 0.25],  # Quarter notes
            accents=[True, False]
        )

        # Syncopated funk pattern:
        RhythmPattern(
            onsets=[0.0, 0.75, 1.5, 2.5, 3.25],
            durations=[0.25, 0.25, 0.5, 0.25, 0.5],
            accents=[True, False, True, False, True]
        )
    """
    onsets: List[float]            # Beat positions (0.0 = downbeat)
    durations: List[float]         # Note durations in beats
    accents: List[bool] = None     # Accent pattern
    velocities: List[int] = None   # MIDI velocities (1-127)
    articulations: List[ArticulationType] = None  # Per-note articulation

    def __post_init__(self):
        """Validate and fill defaults"""
        if len(self.onsets) != len(self.durations):
            raise ValueError("onsets and durations must have same length")

        if self.accents is None:
            self.accents = [False] * len(self.onsets)
        if self.velocities is None:
            # Default: accents = 100, normal = 80
            self.velocities = [100 if acc else 80 for acc in self.accents]
        if self.articulations is None:
            self.articulations = [ArticulationType.STACCATO] * len(self.onsets)

    @property
    def num_events(self) -> int:
        """Number of rhythmic events"""
        return len(self.onsets)

    def apply_swing(self, swing_factor: float = 0.67) -> 'RhythmPattern':
        """
        Apply swing feel to straight 8th notes

        Args:
            swing_factor: 0.5 = straight, 0.67 = triplet swing
        """
        new_onsets = []
        for onset in self.onsets:
            beat = int(onset)
            subdivision = onset - beat

            # Apply swing to 8th notes on offbeats
            if abs(subdivision - 0.5) < 0.01:  # 8th note offbeat
                new_onset = beat + swing_factor
            else:
                new_onset = onset
            new_onsets.append(new_onset)

        return RhythmPattern(
            onsets=new_onsets,
            durations=self.durations.copy(),
            accents=self.accents.copy(),
            velocities=self.velocities.copy(),
            articulations=self.articulations.copy()
        )


@dataclass
class InstrumentAssignment:
    """Assignment of instrument(s) to a rhythm pattern"""
    instrument_names: List[str]    # e.g., ["trumpet", "trombone"]
    rhythm_pattern: RhythmPattern
    chord_progression: List[str]   # Chord symbols (e.g., ["Cmaj7", "Dm7"])
    voicing_strategy: VoicingStrategy = VoicingStrategy.CLOSE
    register_preference: Register = Register.NEUTRAL
    section: InstrumentSection = InstrumentSection.BRASS


@dataclass
class GeneratedNote:
    """A single generated note with all parameters"""
    pitch: int                     # MIDI note number (sounding)
    onset: float                   # Beat position
    duration: float                # Duration in beats
    velocity: int                  # MIDI velocity
    articulation: ArticulationType
    instrument: str                # Instrument name
    written_pitch: Optional[int] = None  # For transposing instruments


@dataclass
class SectionOutput:
    """Output for an instrument section"""
    section: InstrumentSection
    notes: List[GeneratedNote]
    voicing_quality: Playability
    warnings: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


# ============================================================================
# ARTICULATION PATTERN LIBRARY
# ============================================================================

class ArticulationLibrary:
    """Library of idiomatic articulation patterns for each section"""

    # Brass articulation patterns
    BRASS_PATTERNS = {
        'hits': [ArticulationType.TONGUED, ArticulationType.ACCENT],
        'stabs': [ArticulationType.MARCATO, ArticulationType.STACCATO],
        'fall_offs': [ArticulationType.TONGUED, ArticulationType.FALL_OFF],
        'sustained': [ArticulationType.LEGATO, ArticulationType.SLURRED],
        'fast_repeated': [ArticulationType.DOUBLE_TONGUE, ArticulationType.STACCATO],
        'triplets': [ArticulationType.TRIPLE_TONGUE, ArticulationType.ACCENT],
        'shake': [ArticulationType.SHAKE, ArticulationType.TENUTO],
        'jazz_articulation': [ArticulationType.TONGUED, ArticulationType.ACCENT,
                             ArticulationType.SLURRED, ArticulationType.FALL_OFF]
    }

    # String articulation patterns
    STRING_PATTERNS = {
        'short': [ArticulationType.SPICCATO, ArticulationType.STACCATO],
        'accented': [ArticulationType.MARTELE, ArticulationType.ACCENT],
        'smooth': [ArticulationType.LEGATO, ArticulationType.DETACHE],
        'plucked': [ArticulationType.PIZZICATO],
        'tremolo': [ArticulationType.TREMOLO],
        'percussive': [ArticulationType.COL_LEGNO, ArticulationType.STACCATO],
        'metallic': [ArticulationType.SUL_PONTICELLO],
        'classical': [ArticulationType.DETACHE, ArticulationType.LEGATO]
    }

    # Woodwind articulation patterns
    WOODWIND_PATTERNS = {
        'staccato': [ArticulationType.STACCATO, ArticulationType.TONGUED],
        'legato': [ArticulationType.LEGATO, ArticulationType.SLURRED],
        'accented': [ArticulationType.ACCENT, ArticulationType.TONGUED],
        'flutter': [ArticulationType.FLUTTER_TONGUE],
        'slap': [ArticulationType.SLAP_TONGUE, ArticulationType.ACCENT]
    }

    @classmethod
    def get_pattern(cls, section: InstrumentSection, style: str) -> List[ArticulationType]:
        """
        Get articulation pattern for section and style

        Args:
            section: Instrument section
            style: Pattern name (e.g., 'hits', 'smooth', 'staccato')

        Returns:
            List of articulation types to cycle through
        """
        if section == InstrumentSection.BRASS:
            return cls.BRASS_PATTERNS.get(style, [ArticulationType.TONGUED])
        elif section == InstrumentSection.STRINGS:
            return cls.STRING_PATTERNS.get(style, [ArticulationType.DETACHE])
        elif section == InstrumentSection.WOODWINDS:
            return cls.WOODWIND_PATTERNS.get(style, [ArticulationType.TONGUED])
        else:
            return [ArticulationType.STACCATO]

    @classmethod
    def recommend_for_rhythm(cls, section: InstrumentSection,
                           rhythm: RhythmPattern) -> List[ArticulationType]:
        """
        Recommend articulation based on rhythm characteristics

        Args:
            section: Instrument section
            rhythm: Rhythm pattern to analyze

        Returns:
            Recommended articulation pattern
        """
        avg_duration = sum(rhythm.durations) / len(rhythm.durations)

        # Short notes
        if avg_duration < 0.5:
            if section == InstrumentSection.BRASS:
                # Fast repeated notes need double/triple tongue
                if cls._has_fast_repeats(rhythm):
                    return cls.BRASS_PATTERNS['fast_repeated']
                return cls.BRASS_PATTERNS['hits']
            elif section == InstrumentSection.STRINGS:
                return cls.STRING_PATTERNS['short']
            elif section == InstrumentSection.WOODWINDS:
                return cls.WOODWIND_PATTERNS['staccato']

        # Long notes
        elif avg_duration > 2.0:
            if section == InstrumentSection.BRASS:
                return cls.BRASS_PATTERNS['sustained']
            elif section == InstrumentSection.STRINGS:
                return cls.STRING_PATTERNS['smooth']
            elif section == InstrumentSection.WOODWINDS:
                return cls.WOODWIND_PATTERNS['legato']

        # Medium notes - default
        if section == InstrumentSection.BRASS:
            return cls.BRASS_PATTERNS['hits']
        elif section == InstrumentSection.STRINGS:
            return cls.STRING_PATTERNS['classical']
        elif section == InstrumentSection.WOODWINDS:
            return cls.WOODWIND_PATTERNS['staccato']

        return [ArticulationType.STACCATO]

    @staticmethod
    def _has_fast_repeats(rhythm: RhythmPattern) -> bool:
        """Check if rhythm has fast repeated notes"""
        for i in range(len(rhythm.onsets) - 1):
            interval = rhythm.onsets[i + 1] - rhythm.onsets[i]
            if interval < 0.25:  # Faster than 16th notes
                return True
        return False


# ============================================================================
# CHORD-TO-PITCH MAPPING
# ============================================================================

class ChordToPitchMapper:
    """Maps rhythm patterns to pitches based on chord progressions"""

    # Chord tone priority for different contexts
    CHORD_TONE_WEIGHTS = {
        'root_position': {'root': 0.4, 'third': 0.3, 'fifth': 0.2, 'seventh': 0.1},
        'bass_emphasis': {'root': 0.7, 'fifth': 0.2, 'third': 0.1},
        'melody': {'third': 0.4, 'seventh': 0.3, 'root': 0.2, 'fifth': 0.1},
        'inner_voice': {'third': 0.35, 'fifth': 0.35, 'seventh': 0.2, 'root': 0.1}
    }

    @staticmethod
    def parse_chord(chord_symbol: str) -> Dict[str, any]:
        """
        Parse chord symbol to extract tones

        Args:
            chord_symbol: e.g., "Cmaj7", "Dm7b5", "G7#9"

        Returns:
            {
                'root': 0,  # Pitch class
                'quality': 'maj7',
                'chord_tones': [0, 4, 7, 11],  # Pitch classes
                'extensions': [14],  # For #9, etc.
            }
        """
        # Simple parser - can be replaced with more sophisticated one
        root_map = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

        # Extract root
        root_name = chord_symbol[0]
        if len(chord_symbol) > 1 and chord_symbol[1] in ['#', 'b']:
            root_name += chord_symbol[1]
            quality_start = 2
        else:
            quality_start = 1

        root = root_map.get(root_name[0], 0)
        if len(root_name) > 1:
            if root_name[1] == '#':
                root += 1
            elif root_name[1] == 'b':
                root -= 1
        root = root % 12

        # Extract quality
        quality = chord_symbol[quality_start:]

        # Build chord tones based on quality
        chord_tones = [root]

        if 'maj7' in quality or 'M7' in quality:
            chord_tones.extend([(root + 4) % 12, (root + 7) % 12, (root + 11) % 12])
        elif 'm7b5' in quality or 'ø' in quality:
            chord_tones.extend([(root + 3) % 12, (root + 6) % 12, (root + 10) % 12])
        elif 'dim7' in quality or '°7' in quality:
            chord_tones.extend([(root + 3) % 12, (root + 6) % 12, (root + 9) % 12])
        elif 'm7' in quality or 'min7' in quality:
            chord_tones.extend([(root + 3) % 12, (root + 7) % 12, (root + 10) % 12])
        elif '7' in quality:
            chord_tones.extend([(root + 4) % 12, (root + 7) % 12, (root + 10) % 12])
        elif 'maj' in quality or 'M' in quality:
            chord_tones.extend([(root + 4) % 12, (root + 7) % 12])
        elif 'm' in quality or 'min' in quality:
            chord_tones.extend([(root + 3) % 12, (root + 7) % 12])
        else:  # Default to major triad
            chord_tones.extend([(root + 4) % 12, (root + 7) % 12])

        # Extensions
        extensions = []
        if '9' in quality:
            extensions.append((root + 14) % 12)
        if '11' in quality:
            extensions.append((root + 17) % 12)
        if '13' in quality:
            extensions.append((root + 21) % 12)

        return {
            'root': root,
            'quality': quality,
            'chord_tones': chord_tones,
            'extensions': extensions
        }

    @classmethod
    def rhythm_to_notes(cls, rhythm: RhythmPattern, chord: str,
                       target_register: Tuple[int, int],
                       voice_role: str = 'melody') -> List[int]:
        """
        Convert rhythm pattern to pitches based on chord

        Args:
            rhythm: Rhythm pattern
            chord: Chord symbol
            target_register: (min_midi, max_midi) range
            voice_role: 'melody', 'bass', 'inner_voice', 'root_position'

        Returns:
            List of MIDI pitches for each rhythmic event
        """
        chord_info = cls.parse_chord(chord)
        chord_tones = chord_info['chord_tones']

        # Generate pitches in target register
        min_pitch, max_pitch = target_register

        # Get all chord tones in register
        available_pitches = []
        for octave in range(0, 10):
            base = octave * 12
            for tone in chord_tones:
                pitch = base + tone
                if min_pitch <= pitch <= max_pitch:
                    available_pitches.append(pitch)

        if not available_pitches:
            # Fallback: use middle C region
            available_pitches = [60 + tone for tone in chord_tones]

        # Select pitches for each event
        pitches = []
        for i, accent in enumerate(rhythm.accents):
            if accent:
                # Accented notes: use root or third
                preferred = [p for p in available_pitches
                           if p % 12 in [chord_info['root'],
                                        (chord_info['root'] + 4) % 12,
                                        (chord_info['root'] + 3) % 12]]
                pitch = random.choice(preferred if preferred else available_pitches)
            else:
                # Unaccented: any chord tone
                pitch = random.choice(available_pitches)

            pitches.append(pitch)

        return pitches


# ============================================================================
# SECTION-SPECIFIC VOICING ENGINES
# ============================================================================

class BrassVoicingEngine:
    """Brass section voicing with idiomatic considerations"""

    # Typical brass section instruments
    INSTRUMENTS = {
        'big_band': ['trumpet', 'trumpet', 'trumpet', 'trumpet',
                     'trombone', 'trombone', 'trombone', 'bass_trombone'],
        'brass_quartet': ['trumpet', 'trumpet', 'trombone', 'tuba'],
        'brass_quintet': ['trumpet', 'trumpet', 'horn', 'trombone', 'tuba'],
        'fanfare': ['trumpet', 'trumpet', 'trumpet', 'horn', 'trombone'],
    }

    @classmethod
    def voice_chord(cls, chord: str, ensemble: str = 'big_band',
                   voicing_type: VoicingStrategy = VoicingStrategy.DROP_2) -> List[Tuple[str, int]]:
        """
        Voice chord for brass section

        Args:
            chord: Chord symbol
            ensemble: 'big_band', 'brass_quartet', etc.
            voicing_type: Voicing strategy

        Returns:
            [(instrument, midi_pitch), ...]
        """
        instruments = cls.INSTRUMENTS.get(ensemble, cls.INSTRUMENTS['big_band'])
        chord_info = ChordToPitchMapper.parse_chord(chord)

        # Get chord tones
        chord_tones = chord_info['chord_tones']
        root = chord_info['root']

        # Build voicing based on strategy
        if voicing_type == VoicingStrategy.DROP_2:
            # Classic big band drop-2 voicing
            # Top voice = melody (lead trumpet)
            # Drop second voice down an octave
            top_note = 72 + chord_tones[0]  # Root in 5th octave

            voicing_pcs = [
                (chord_tones[0], 0),   # Root
                (chord_tones[1], 0),   # Third
                (chord_tones[2], 0),   # Fifth
                (chord_tones[3] if len(chord_tones) > 3 else chord_tones[0], 0)
            ]

            # Assign to instruments (top to bottom)
            result = []
            for i, inst in enumerate(instruments[:4]):
                pc, octave = voicing_pcs[i]
                pitch = 60 + pc + (3 - i) * 7  # Spread over range
                result.append((inst, pitch))

            return result

        elif voicing_type == VoicingStrategy.UNISON:
            # All instruments play same pitch
            pitch = 67  # G4 - good for brass unison
            return [(inst, pitch) for inst in instruments]

        elif voicing_type == VoicingStrategy.OCTAVES:
            # Octave doublings
            base_pitch = 60 + root
            result = []
            for i, inst in enumerate(instruments):
                octave = i // 2  # Two instruments per octave
                pitch = base_pitch + octave * 12
                result.append((inst, pitch))
            return result

        else:  # Default close voicing
            base = 60 + root
            result = []
            for i, inst in enumerate(instruments[:len(chord_tones)]):
                pitch = base + chord_tones[i % len(chord_tones)] + (i // len(chord_tones)) * 12
                result.append((inst, pitch))
            return result


class StringVoicingEngine:
    """String section voicing with bowing and articulation"""

    INSTRUMENTS = {
        'string_quartet': ['violin', 'violin', 'viola', 'cello'],
        'string_section': ['violin', 'violin', 'violin', 'violin',
                          'viola', 'viola', 'cello', 'cello', 'double_bass'],
        'chamber': ['violin', 'viola', 'cello']
    }

    @classmethod
    def voice_chord(cls, chord: str, ensemble: str = 'string_quartet',
                   voicing_type: VoicingStrategy = VoicingStrategy.CLOSE,
                   technique: StringTechnique = StringTechnique.ARCO) -> List[Tuple[str, int]]:
        """
        Voice chord for string section

        Args:
            chord: Chord symbol
            ensemble: 'string_quartet', 'string_section', 'chamber'
            voicing_type: Voicing strategy
            technique: String technique to use

        Returns:
            [(instrument, midi_pitch), ...]
        """
        instruments = cls.INSTRUMENTS.get(ensemble, cls.INSTRUMENTS['string_quartet'])
        chord_info = ChordToPitchMapper.parse_chord(chord)
        chord_tones = chord_info['chord_tones']

        # String quartet: standard SATB-like spacing
        if ensemble == 'string_quartet':
            # Violin 1: melody (highest)
            # Violin 2: alto
            # Viola: tenor
            # Cello: bass

            result = []
            base_pitches = [72, 67, 62, 48]  # Typical ranges

            for i, inst in enumerate(instruments):
                pc = chord_tones[i % len(chord_tones)]
                pitch = base_pitches[i] + (pc - (base_pitches[i] % 12))
                # Adjust to nearest chord tone
                while pitch % 12 not in chord_tones:
                    pitch += 1
                result.append((inst, pitch))

            return result

        elif voicing_type == VoicingStrategy.UNISON:
            # Unison - powerful effect
            pitch = 60 + chord_tones[0]
            return [(inst, pitch) for inst in instruments]

        else:  # Open voicing
            result = []
            base = 48
            for i, inst in enumerate(instruments):
                pc = chord_tones[i % len(chord_tones)]
                octave = i // len(chord_tones)
                pitch = base + pc + octave * 12
                result.append((inst, pitch))
            return result


class WoodwindVoicingEngine:
    """Woodwind section voicing"""

    INSTRUMENTS = {
        'wind_quartet': ['flute', 'oboe', 'clarinet', 'bassoon'],
        'wind_quintet': ['flute', 'oboe', 'clarinet', 'horn', 'bassoon'],
        'symphonic': ['flute', 'flute', 'oboe', 'oboe',
                     'clarinet', 'clarinet', 'bassoon', 'bassoon']
    }

    @classmethod
    def voice_chord(cls, chord: str, ensemble: str = 'wind_quartet',
                   voicing_type: VoicingStrategy = VoicingStrategy.TRADITIONAL) -> List[Tuple[str, int]]:
        """
        Voice chord for woodwind section

        Args:
            chord: Chord symbol
            ensemble: 'wind_quartet', 'wind_quintet', 'symphonic'
            voicing_type: Traditional (flute top) or interlocking

        Returns:
            [(instrument, midi_pitch), ...]
        """
        instruments = cls.INSTRUMENTS.get(ensemble, cls.INSTRUMENTS['wind_quartet'])
        chord_info = ChordToPitchMapper.parse_chord(chord)
        chord_tones = chord_info['chord_tones']

        if voicing_type == VoicingStrategy.TRADITIONAL:
            # Traditional: flute highest, then oboe, clarinet, bassoon
            # Respect individual instrument strengths

            base_pitches = {
                'flute': 72,      # Bright, clear in this range
                'oboe': 67,       # Sweet, penetrating
                'clarinet': 62,   # Rich, warm
                'bassoon': 48,    # Dark, woody
                'horn': 55        # Mellow brass color
            }

            result = []
            for inst in instruments:
                base = base_pitches.get(inst, 60)
                # Find nearest chord tone
                pc = chord_tones[len(result) % len(chord_tones)]
                pitch = base + (pc - (base % 12))
                while pitch % 12 not in chord_tones:
                    pitch += 1
                result.append((inst, pitch))

            return result

        elif voicing_type == VoicingStrategy.INTERLOCKING:
            # Interlocking: mix timbres for blended sound
            result = []
            base = 60

            for i, inst in enumerate(instruments):
                pc = chord_tones[i % len(chord_tones)]
                pitch = base + pc + (i // len(chord_tones)) * 12
                result.append((inst, pitch))

            # Sort by pitch to create interlocking
            result.sort(key=lambda x: x[1])
            return result

        else:  # Close voicing
            base = 60
            result = []
            for i, inst in enumerate(instruments[:len(chord_tones)]):
                pitch = base + chord_tones[i]
                result.append((inst, pitch))
            return result


# ============================================================================
# MAIN GRANULAR CONTROL ENGINE
# ============================================================================

class GranularControl:
    """
    Main engine for granular control of musical components

    Usage:
        # Create brass hits on beats 1 and 3
        gc = GranularControl()

        rhythm = RhythmPattern(
            onsets=[0.0, 2.0],
            durations=[0.25, 0.25],
            accents=[True, True]
        )

        output = gc.generate(
            rhythm_pattern=rhythm,
            chord_progression=["Cmaj7", "Dm7", "G7", "Cmaj7"],
            section=InstrumentSection.BRASS,
            instruments=['trumpet', 'trombone'],
            voicing_strategy=VoicingStrategy.DROP_2,
            articulation_style='hits'
        )
    """

    def __init__(self):
        """Initialize granular control engine"""
        self.articulation_lib = ArticulationLibrary()
        self.chord_mapper = ChordToPitchMapper()

        # Voicing engines
        self.brass_voicing = BrassVoicingEngine()
        self.string_voicing = StringVoicingEngine()
        self.woodwind_voicing = WoodwindVoicingEngine()

    def generate(self,
                rhythm_pattern: RhythmPattern,
                chord_progression: List[str],
                section: InstrumentSection,
                instruments: List[str] = None,
                voicing_strategy: VoicingStrategy = VoicingStrategy.CLOSE,
                articulation_style: str = None,
                register_preference: Register = Register.NEUTRAL,
                measures: int = 4,
                beats_per_measure: int = 4,
                apply_swing: bool = False,
                swing_factor: float = 0.67) -> SectionOutput:
        """
        Generate musical notation for given parameters

        Args:
            rhythm_pattern: User-defined rhythm
            chord_progression: List of chord symbols
            section: Instrument section type
            instruments: Specific instruments (None = use defaults)
            voicing_strategy: How to voice chords
            articulation_style: Style name or None for automatic
            register_preference: Preferred register
            measures: Number of measures to generate
            beats_per_measure: Time signature numerator
            apply_swing: Apply swing feel
            swing_factor: Swing amount (0.5-0.67)

        Returns:
            SectionOutput with generated notes
        """
        # Apply swing if requested
        if apply_swing:
            rhythm_pattern = rhythm_pattern.apply_swing(swing_factor)

        # Get or recommend articulations
        if articulation_style:
            articulations = self.articulation_lib.get_pattern(section, articulation_style)
        else:
            articulations = self.articulation_lib.recommend_for_rhythm(section, rhythm_pattern)

        # Generate notes
        generated_notes = []
        warnings = []

        # Loop through measures
        for measure in range(measures):
            chord = chord_progression[measure % len(chord_progression)]
            measure_offset = measure * beats_per_measure

            # Get voicing for this chord
            voicing = self._voice_chord(chord, section, instruments, voicing_strategy)

            # Generate notes for each rhythmic event
            for event_idx in range(rhythm_pattern.num_events):
                onset = measure_offset + rhythm_pattern.onsets[event_idx]
                duration = rhythm_pattern.durations[event_idx]
                velocity = rhythm_pattern.velocities[event_idx]

                # Get articulation (cycle through pattern)
                artic = articulations[event_idx % len(articulations)]

                # Generate note for each instrument in voicing
                for inst_name, pitch in voicing:
                    # Validate range
                    if inst_name in INSTRUMENT_DATABASE:
                        inst_info = INSTRUMENT_DATABASE[inst_name]
                        if not (inst_info.lowest_note <= pitch <= inst_info.highest_note):
                            warnings.append(
                                f"{inst_name}: pitch {pitch} outside range "
                                f"({inst_info.lowest_note}-{inst_info.highest_note})"
                            )
                            # Transpose to valid range
                            while pitch < inst_info.lowest_note:
                                pitch += 12
                            while pitch > inst_info.highest_note:
                                pitch -= 12

                    note = GeneratedNote(
                        pitch=pitch,
                        onset=onset,
                        duration=duration,
                        velocity=velocity,
                        articulation=artic,
                        instrument=inst_name
                    )
                    generated_notes.append(note)

        # Assess playability
        playability = self._assess_playability(generated_notes, section)

        return SectionOutput(
            section=section,
            notes=generated_notes,
            voicing_quality=playability,
            warnings=warnings,
            suggestions=self._generate_suggestions(generated_notes, section)
        )

    def _voice_chord(self, chord: str, section: InstrumentSection,
                    instruments: Optional[List[str]],
                    strategy: VoicingStrategy) -> List[Tuple[str, int]]:
        """Voice a single chord for the section"""

        if section == InstrumentSection.BRASS:
            ensemble = 'big_band' if not instruments else 'custom'
            if ensemble == 'custom':
                # Custom voicing
                chord_info = ChordToPitchMapper.parse_chord(chord)
                result = []
                base = 60
                for i, inst in enumerate(instruments):
                    pc = chord_info['chord_tones'][i % len(chord_info['chord_tones'])]
                    pitch = base + pc + (i // len(chord_info['chord_tones'])) * 12
                    result.append((inst, pitch))
                return result
            else:
                return self.brass_voicing.voice_chord(chord, ensemble, strategy)

        elif section == InstrumentSection.STRINGS:
            ensemble = 'string_quartet' if not instruments else 'custom'
            if ensemble == 'custom':
                chord_info = ChordToPitchMapper.parse_chord(chord)
                result = []
                base = 55
                for i, inst in enumerate(instruments):
                    pc = chord_info['chord_tones'][i % len(chord_info['chord_tones'])]
                    pitch = base + pc + (i // len(chord_info['chord_tones'])) * 12
                    result.append((inst, pitch))
                return result
            else:
                return self.string_voicing.voice_chord(chord, ensemble, strategy)

        elif section == InstrumentSection.WOODWINDS:
            ensemble = 'wind_quartet' if not instruments else 'custom'
            if ensemble == 'custom':
                chord_info = ChordToPitchMapper.parse_chord(chord)
                result = []
                base = 62
                for i, inst in enumerate(instruments):
                    pc = chord_info['chord_tones'][i % len(chord_info['chord_tones'])]
                    pitch = base + pc + (i // len(chord_info['chord_tones'])) * 12
                    result.append((inst, pitch))
                return result
            else:
                return self.woodwind_voicing.voice_chord(chord, ensemble, strategy)

        else:
            # Generic voicing
            if not instruments:
                instruments = ['piano']

            chord_info = ChordToPitchMapper.parse_chord(chord)
            result = []
            base = 60
            for i, inst in enumerate(instruments):
                pc = chord_info['chord_tones'][i % len(chord_info['chord_tones'])]
                pitch = base + pc
                result.append((inst, pitch))
            return result

    def _assess_playability(self, notes: List[GeneratedNote],
                          section: InstrumentSection) -> Playability:
        """Assess overall playability of generated part"""

        # Check for range violations
        range_violations = sum(1 for note in notes
                             if note.instrument in INSTRUMENT_DATABASE
                             and not (INSTRUMENT_DATABASE[note.instrument].lowest_note
                                    <= note.pitch
                                    <= INSTRUMENT_DATABASE[note.instrument].highest_note))

        if range_violations > len(notes) * 0.3:
            return Playability.PROBLEMATIC
        elif range_violations > len(notes) * 0.1:
            return Playability.ACCEPTABLE

        # Check for awkward intervals (large leaps)
        leap_issues = 0
        for i in range(len(notes) - 1):
            if notes[i].instrument == notes[i+1].instrument:
                interval = abs(notes[i+1].pitch - notes[i].pitch)
                if interval > 12:  # Octave leap
                    leap_issues += 1

        if leap_issues > len(notes) * 0.2:
            return Playability.DIFFICULT

        return Playability.EXCELLENT

    def _generate_suggestions(self, notes: List[GeneratedNote],
                            section: InstrumentSection) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []

        # Analyze register usage
        pitches = [n.pitch for n in notes]
        if pitches:
            avg_pitch = sum(pitches) / len(pitches)

            if section == InstrumentSection.BRASS:
                if avg_pitch > 80:
                    suggestions.append("Consider lowering register for more comfortable brass writing")
                elif avg_pitch < 50:
                    suggestions.append("Register may be too low for brass clarity")

            elif section == InstrumentSection.STRINGS:
                if avg_pitch < 45:
                    suggestions.append("Very low register - consider double bass or cello solo")

        return suggestions

    def generate_hits(self, rhythm: RhythmPattern, chord_progression: List[str],
                     section: InstrumentSection = InstrumentSection.BRASS,
                     **kwargs) -> SectionOutput:
        """
        Convenience method for generating "hits" (accented, short notes)

        Automatically configures for hit-style writing:
        - Staccato articulation
        - Accented
        - Appropriate voicing for impact
        """
        kwargs['articulation_style'] = 'hits' if section == InstrumentSection.BRASS else 'accented'
        kwargs['voicing_strategy'] = kwargs.get('voicing_strategy', VoicingStrategy.DROP_2)

        return self.generate(
            rhythm_pattern=rhythm,
            chord_progression=chord_progression,
            section=section,
            **kwargs
        )

    def generate_sustained(self, rhythm: RhythmPattern, chord_progression: List[str],
                          section: InstrumentSection = InstrumentSection.STRINGS,
                          **kwargs) -> SectionOutput:
        """
        Convenience method for sustained pad-like writing
        """
        kwargs['articulation_style'] = 'sustained' if section == InstrumentSection.BRASS else 'smooth'
        kwargs['voicing_strategy'] = kwargs.get('voicing_strategy', VoicingStrategy.CLOSE)

        return self.generate(
            rhythm_pattern=rhythm,
            chord_progression=chord_progression,
            section=section,
            **kwargs
        )

    def to_midi(self, output: SectionOutput, filename: str, tempo: int = 120):
        """
        Export generated output to MIDI file

        Args:
            output: Generated section output
            filename: Output MIDI file path
            tempo: Tempo in BPM
        """
        try:
            import mido
            from mido import Message, MidiFile, MidiTrack, MetaMessage
        except ImportError:
            print("Warning: mido library not available. Cannot export MIDI.")
            return

        mid = MidiFile()

        # Create track per instrument
        instruments_used = set(note.instrument for note in output.notes)

        for inst in instruments_used:
            track = MidiTrack()
            mid.tracks.append(track)

            # Set tempo
            track.append(MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo), time=0))

            # Track name
            track.append(MetaMessage('track_name', name=inst, time=0))

            # Get notes for this instrument
            inst_notes = [n for n in output.notes if n.instrument == inst]
            inst_notes.sort(key=lambda n: n.onset)

            # Convert to MIDI messages
            current_time = 0
            ticks_per_beat = 480

            for note in inst_notes:
                # Note on
                onset_ticks = int(note.onset * ticks_per_beat)
                delta_time = onset_ticks - current_time

                track.append(Message('note_on', note=note.pitch,
                                   velocity=note.velocity, time=delta_time))
                current_time = onset_ticks

                # Note off
                duration_ticks = int(note.duration * ticks_per_beat)
                track.append(Message('note_off', note=note.pitch,
                                   velocity=0, time=duration_ticks))
                current_time += duration_ticks

        mid.save(filename)
        print(f"MIDI file saved: {filename}")


# ============================================================================
# PERCUSSION AND RHYTHM SECTION SUPPORT
# ============================================================================

class PercussionVoicingEngine:
    """Percussion and drum pattern generation"""

    # General MIDI drum map
    DRUM_MAP = {
        'kick': 36,
        'snare': 38,
        'closed_hihat': 42,
        'open_hihat': 46,
        'crash': 49,
        'ride': 51,
        'tom_low': 45,
        'tom_mid': 47,
        'tom_high': 50,
        'cowbell': 56,
        'tambourine': 54,
        'clap': 39,
        'shaker': 70,
        'conga_low': 64,
        'conga_high': 62,
        'bongo_low': 61,
        'bongo_high': 60
    }

    @classmethod
    def rhythm_to_drums(cls, rhythm: RhythmPattern,
                       drum_voices: List[str] = None) -> List[GeneratedNote]:
        """
        Convert rhythm pattern to drum hits

        Args:
            rhythm: Rhythm pattern
            drum_voices: Drum voice names (e.g., ['kick', 'snare'])

        Returns:
            List of drum notes
        """
        if drum_voices is None:
            drum_voices = ['snare']  # Default to snare

        notes = []

        for i in range(rhythm.num_events):
            # Cycle through drum voices
            drum_voice = drum_voices[i % len(drum_voices)]
            pitch = cls.DRUM_MAP.get(drum_voice, 38)  # Default to snare

            note = GeneratedNote(
                pitch=pitch,
                onset=rhythm.onsets[i],
                duration=rhythm.durations[i],
                velocity=rhythm.velocities[i],
                articulation=rhythm.articulations[i],
                instrument='drums'
            )
            notes.append(note)

        return notes

    @classmethod
    def create_basic_beat(cls, style: str = 'rock',
                         measures: int = 4,
                         beats_per_measure: int = 4) -> List[GeneratedNote]:
        """
        Create basic drum beat

        Args:
            style: 'rock', 'jazz', 'funk', 'latin'
            measures: Number of measures
            beats_per_measure: Time signature numerator

        Returns:
            Drum notes
        """
        notes = []

        if style == 'rock':
            # Rock beat: kick on 1 and 3, snare on 2 and 4, hihats on all 8ths
            for m in range(measures):
                offset = m * beats_per_measure

                # Kick on 1 and 3
                for beat in [0, 2]:
                    notes.append(GeneratedNote(
                        pitch=cls.DRUM_MAP['kick'],
                        onset=offset + beat,
                        duration=0.25,
                        velocity=100,
                        articulation=ArticulationType.ACCENT,
                        instrument='drums'
                    ))

                # Snare on 2 and 4
                for beat in [1, 3]:
                    notes.append(GeneratedNote(
                        pitch=cls.DRUM_MAP['snare'],
                        onset=offset + beat,
                        duration=0.25,
                        velocity=100,
                        articulation=ArticulationType.ACCENT,
                        instrument='drums'
                    ))

                # Hihats on 8th notes
                for eighth in range(beats_per_measure * 2):
                    notes.append(GeneratedNote(
                        pitch=cls.DRUM_MAP['closed_hihat'],
                        onset=offset + eighth * 0.5,
                        duration=0.25,
                        velocity=70,
                        articulation=ArticulationType.STACCATO,
                        instrument='drums'
                    ))

        elif style == 'jazz':
            # Jazz ride pattern with swing
            for m in range(measures):
                offset = m * beats_per_measure

                # Ride cymbal (swing)
                for beat in range(beats_per_measure):
                    # Downbeat
                    notes.append(GeneratedNote(
                        pitch=cls.DRUM_MAP['ride'],
                        onset=offset + beat,
                        duration=0.33,
                        velocity=80,
                        articulation=ArticulationType.ACCENT,
                        instrument='drums'
                    ))
                    # Upbeat (swung)
                    notes.append(GeneratedNote(
                        pitch=cls.DRUM_MAP['ride'],
                        onset=offset + beat + 0.67,
                        duration=0.33,
                        velocity=60,
                        articulation=ArticulationType.STACCATO,
                        instrument='drums'
                    ))

                # Snare on 2 and 4 (comping)
                for beat in [1, 3]:
                    if random.random() > 0.3:  # Add variety
                        notes.append(GeneratedNote(
                            pitch=cls.DRUM_MAP['snare'],
                            onset=offset + beat + random.choice([0, 0.25, 0.5]),
                            duration=0.1,
                            velocity=random.randint(60, 90),
                            articulation=ArticulationType.STACCATO,
                            instrument='drums'
                        ))

        return notes


# ============================================================================
# DYNAMICS AND EXPRESSION
# ============================================================================

class DynamicsEngine:
    """Dynamic shaping and expression control"""

    # Dynamic levels (MIDI velocity)
    DYNAMICS = {
        'ppp': 20,
        'pp': 35,
        'p': 50,
        'mp': 65,
        'mf': 80,
        'f': 95,
        'ff': 110,
        'fff': 127
    }

    @classmethod
    def apply_dynamics_curve(cls, notes: List[GeneratedNote],
                            curve_type: str = 'crescendo',
                            start_dynamic: str = 'p',
                            end_dynamic: str = 'f') -> List[GeneratedNote]:
        """
        Apply dynamic curve to notes

        Args:
            notes: Notes to modify
            curve_type: 'crescendo', 'decrescendo', 'swell', 'arch'
            start_dynamic: Starting dynamic level
            end_dynamic: Ending dynamic level

        Returns:
            Modified notes
        """
        start_vel = cls.DYNAMICS.get(start_dynamic, 65)
        end_vel = cls.DYNAMICS.get(end_dynamic, 95)

        for i, note in enumerate(notes):
            ratio = i / (len(notes) - 1) if len(notes) > 1 else 0

            if curve_type == 'crescendo':
                velocity = int(start_vel + (end_vel - start_vel) * ratio)
            elif curve_type == 'decrescendo':
                velocity = int(start_vel - (start_vel - end_vel) * ratio)
            elif curve_type == 'swell':
                # Crescendo then decrescendo
                if ratio < 0.5:
                    velocity = int(start_vel + (end_vel - start_vel) * (ratio * 2))
                else:
                    velocity = int(end_vel - (end_vel - start_vel) * ((ratio - 0.5) * 2))
            elif curve_type == 'arch':
                # Bell curve
                import math
                velocity = int(start_vel + (end_vel - start_vel) * math.sin(ratio * math.pi))
            else:
                velocity = note.velocity

            note.velocity = max(1, min(127, velocity))

        return notes

    @classmethod
    def apply_accents(cls, notes: List[GeneratedNote],
                     accent_pattern: List[bool],
                     accent_amount: int = 20) -> List[GeneratedNote]:
        """
        Apply accent pattern to notes

        Args:
            notes: Notes to modify
            accent_pattern: Boolean pattern (cycles)
            accent_amount: Velocity increase for accents

        Returns:
            Modified notes
        """
        for i, note in enumerate(notes):
            if accent_pattern[i % len(accent_pattern)]:
                note.velocity = min(127, note.velocity + accent_amount)

        return notes


# ============================================================================
# PHRASE SHAPING AND MUSICALITY
# ============================================================================

class PhraseShaper:
    """Musical phrase shaping and articulation refinement"""

    @staticmethod
    def add_phrase_ending(notes: List[GeneratedNote],
                         ending_type: str = 'ritardando') -> List[GeneratedNote]:
        """
        Add musical phrase ending

        Args:
            notes: Notes to modify
            ending_type: 'ritardando', 'fermata', 'breath', 'decrescendo'

        Returns:
            Modified notes
        """
        if not notes:
            return notes

        if ending_type == 'ritardando':
            # Gradually slow down last few notes
            num_affected = min(4, len(notes))
            for i in range(len(notes) - num_affected, len(notes)):
                # Slightly delay onset
                delay_factor = ((i - (len(notes) - num_affected)) / num_affected) * 0.1
                notes[i].onset += delay_factor

        elif ending_type == 'decrescendo':
            # Fade out last notes
            num_affected = min(4, len(notes))
            for i in range(len(notes) - num_affected, len(notes)):
                fade_factor = 1.0 - ((i - (len(notes) - num_affected)) / num_affected) * 0.5
                notes[i].velocity = int(notes[i].velocity * fade_factor)

        elif ending_type == 'breath':
            # Add small gap before last note (breath mark)
            if len(notes) > 1:
                notes[-1].onset += 0.1

        return notes

    @staticmethod
    def add_ornaments(notes: List[GeneratedNote],
                     ornament_type: str = 'mordent',
                     positions: List[int] = None) -> List[GeneratedNote]:
        """
        Add ornaments to specified notes

        Args:
            notes: Base notes
            ornament_type: 'mordent', 'turn', 'trill', 'grace_note'
            positions: Indices of notes to ornament (None = first and last)

        Returns:
            Notes with ornaments added
        """
        if positions is None:
            positions = [0, len(notes) - 1] if len(notes) > 1 else [0]

        new_notes = []

        for i, note in enumerate(notes):
            if i in positions:
                if ornament_type == 'mordent':
                    # Add quick lower neighbor
                    grace = GeneratedNote(
                        pitch=note.pitch - 1,
                        onset=note.onset - 0.05,
                        duration=0.05,
                        velocity=note.velocity - 10,
                        articulation=ArticulationType.STACCATO,
                        instrument=note.instrument
                    )
                    new_notes.append(grace)
                elif ornament_type == 'grace_note':
                    # Add grace note from below
                    grace = GeneratedNote(
                        pitch=note.pitch - 2,
                        onset=note.onset - 0.08,
                        duration=0.08,
                        velocity=note.velocity - 15,
                        articulation=ArticulationType.STACCATO,
                        instrument=note.instrument
                    )
                    new_notes.append(grace)

            new_notes.append(note)

        return new_notes


# ============================================================================
# ADVANCED CONTROL FEATURES
# ============================================================================

class AdvancedControlEngine:
    """Advanced features for granular control"""

    @staticmethod
    def apply_humanization(notes: List[GeneratedNote],
                          timing_variance: float = 0.02,
                          velocity_variance: int = 5) -> List[GeneratedNote]:
        """
        Humanize mechanical MIDI by adding subtle timing and velocity variations

        Args:
            notes: Notes to humanize
            timing_variance: Max timing deviation in beats
            velocity_variance: Max velocity deviation

        Returns:
            Humanized notes
        """
        import random

        for note in notes:
            # Add slight timing variation
            note.onset += random.uniform(-timing_variance, timing_variance)

            # Add velocity variation
            vel_change = random.randint(-velocity_variance, velocity_variance)
            note.velocity = max(1, min(127, note.velocity + vel_change))

        # Sort by onset to maintain order
        notes.sort(key=lambda n: n.onset)

        return notes

    @staticmethod
    def create_layered_texture(base_rhythm: RhythmPattern,
                              chord_progression: List[str],
                              layers: List[Dict]) -> List[SectionOutput]:
        """
        Create multi-layered texture with different sections

        Args:
            base_rhythm: Base rhythm pattern
            chord_progression: Chord progression
            layers: List of layer specifications
                    [{'section': InstrumentSection.BRASS, 'instruments': [...], ...}, ...]

        Returns:
            List of SectionOutput for each layer
        """
        gc = GranularControl()
        outputs = []

        for layer_spec in layers:
            # Create variation of rhythm for this layer
            layer_rhythm = copy.deepcopy(base_rhythm)

            # Apply layer-specific transformations
            if 'offset' in layer_spec:
                # Shift timing
                layer_rhythm.onsets = [o + layer_spec['offset'] for o in layer_rhythm.onsets]

            if 'duration_multiplier' in layer_spec:
                layer_rhythm.durations = [d * layer_spec['duration_multiplier']
                                         for d in layer_rhythm.durations]

            # Generate layer
            output = gc.generate(
                rhythm_pattern=layer_rhythm,
                chord_progression=chord_progression,
                section=layer_spec.get('section', InstrumentSection.BRASS),
                instruments=layer_spec.get('instruments'),
                voicing_strategy=layer_spec.get('voicing_strategy', VoicingStrategy.CLOSE),
                measures=layer_spec.get('measures', 4)
            )

            outputs.append(output)

        return outputs


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_brass_hits(onsets: List[float], chord_progression: List[str],
                     **kwargs) -> SectionOutput:
    """
    Quick function to create brass hits

    Args:
        onsets: Beat positions for hits (e.g., [0.0, 2.0] for 1 and 3)
        chord_progression: Chord symbols
        **kwargs: Additional parameters

    Returns:
        Generated brass hits
    """
    rhythm = RhythmPattern(
        onsets=onsets,
        durations=[0.25] * len(onsets),
        accents=[True] * len(onsets)
    )

    gc = GranularControl()
    return gc.generate_hits(rhythm, chord_progression,
                          section=InstrumentSection.BRASS, **kwargs)


def create_string_pad(duration: float, chord_progression: List[str],
                     **kwargs) -> SectionOutput:
    """
    Quick function to create string pad

    Args:
        duration: Duration of each chord in beats
        chord_progression: Chord symbols
        **kwargs: Additional parameters

    Returns:
        Generated string pad
    """
    onsets = [i * duration for i in range(len(chord_progression))]
    rhythm = RhythmPattern(
        onsets=onsets,
        durations=[duration] * len(onsets),
        accents=[False] * len(onsets)
    )

    gc = GranularControl()
    return gc.generate_sustained(rhythm, chord_progression,
                               section=InstrumentSection.STRINGS, **kwargs)


# ============================================================================
# EXAMPLES AND DEMONSTRATIONS
# ============================================================================

def example_brass_hits():
    """Example: Generate brass hits on beats 1 and 3"""
    print("=" * 60)
    print("Example: Brass Hits on Beats 1 and 3")
    print("=" * 60)

    rhythm = RhythmPattern(
        onsets=[0.0, 2.0],  # Beats 1 and 3
        durations=[0.25, 0.25],
        accents=[True, True]
    )

    chords = ["Cmaj7", "Dm7", "G7", "Cmaj7"]

    gc = GranularControl()
    output = gc.generate_hits(
        rhythm=rhythm,
        chord_progression=chords,
        measures=4,
        instruments=['trumpet', 'trumpet', 'trombone', 'trombone']
    )

    print(f"\nGenerated {len(output.notes)} notes")
    print(f"Playability: {output.playability.name}")

    if output.warnings:
        print("\nWarnings:")
        for w in output.warnings:
            print(f"  - {w}")

    if output.suggestions:
        print("\nSuggestions:")
        for s in output.suggestions:
            print(f"  - {s}")

    # Show first few notes
    print("\nFirst 8 notes:")
    for note in output.notes[:8]:
        print(f"  {note.instrument}: pitch={note.pitch}, onset={note.onset:.2f}, "
              f"dur={note.duration:.2f}, vel={note.velocity}, artic={note.articulation.value}")

    return output


def example_funk_rhythm():
    """Example: Funk rhythm with syncopation"""
    print("\n" + "=" * 60)
    print("Example: Syncopated Funk Brass Pattern")
    print("=" * 60)

    # Syncopated funk pattern
    rhythm = RhythmPattern(
        onsets=[0.0, 0.75, 1.5, 2.5, 3.25],
        durations=[0.25, 0.25, 0.5, 0.25, 0.5],
        accents=[True, False, True, False, True]
    )

    chords = ["Em7", "A7", "Dm7", "G7"]

    gc = GranularControl()
    output = gc.generate(
        rhythm_pattern=rhythm,
        chord_progression=chords,
        section=InstrumentSection.BRASS,
        articulation_style='hits',
        voicing_strategy=VoicingStrategy.DROP_2,
        measures=4
    )

    print(f"\nGenerated {len(output.notes)} notes")
    print(f"Playability: {output.playability.name}")

    return output


if __name__ == "__main__":
    print("Granular Control System - Examples")
    print("=" * 60)

    # Run examples
    output1 = example_brass_hits()
    output2 = example_funk_rhythm()

    print("\n" + "=" * 60)
    print("Examples complete!")
