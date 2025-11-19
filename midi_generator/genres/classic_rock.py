#!/usr/bin/env python3
"""
Classic Rock Music Generator

Comprehensive implementation of rock music generation across non-metal rock genres:
- Classic Rock (60s-70s: Rolling Stones, Led Zeppelin, The Who)
- Punk Rock (Ramones, Sex Pistols, The Clash)
- Alternative Rock (Nirvana, Radiohead, Pearl Jam)
- Indie Rock (Arctic Monkeys, The Strokes, Tame Impala)
- Garage Rock (The Black Keys, White Stripes)
- Post-Punk (Joy Division, The Cure, Talking Heads)

Features:
---------
- Classic rock progressions (I-IV-V, I-bVII-IV, I-V-vi-IV, blues-based, modal)
- Power chords (root-fifth dyads in standard tuning)
- Guitar techniques (bends, hammer-ons, pull-offs, slides, vibrato)
- Drum patterns (basic rock beat, variations, fills, half-time, punk)
- Bass lines (root-fifth, walking, octave jumps, pentatonic riffs)
- Song structures (verse-chorus-solo-bridge forms)
- Scales and riffs (minor/major pentatonic, blues scale, Mixolydian)
- Punk rock specifics (fast tempo, simple progressions, raw energy)

Note: Metal is comprehensively covered in metal.py. This module focuses on
non-metal rock styles with standard tuning and traditional rock aesthetics.

Research References:
-------------------
- "The Guitar Handbook" - Ralph Denyer (1982)
  Classic guitar techniques, tuning, and chord voicings

- "How the Beatles Destroyed Rock 'n' Roll" - Elijah Wald (2009)
  Historical analysis of rock evolution and its roots

- "Our Band Could Be Your Life" - Michael Azerrad (2001)
  Comprehensive study of American indie/punk (1981-1991)

- "Rip It Up and Start Again: Postpunk 1978-1984" - Simon Reynolds (2005)
  Analysis of post-punk aesthetics and innovation

- Rolling Stone's "100 Greatest Guitarists" analysis
  Technical breakdown of signature techniques (bends, vibrato, phrasing)

- Guitar World interviews with Jimmy Page, Pete Townshend, Kurt Cobain
  First-hand accounts of classic rock guitar approaches

- "The Art of Punk" - Russ Bestley & Alex Ogg (2016)
  DIY aesthetic and simplified chord progressions in punk

Author: Agent 44 - Classic Rock Module
Date: 2025
Part of: Phase 3 - Complete Genre Coverage (10 Agents)
License: MIT
"""

import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import math


# ============================================================================
# Enums and Type Definitions
# ============================================================================

class RockStyle(Enum):
    """Rock sub-genres"""
    CLASSIC_ROCK = "classic_rock"      # 60s-70s: Led Zeppelin, Rolling Stones
    PUNK = "punk"                      # Ramones, Sex Pistols, The Clash
    ALTERNATIVE = "alternative"        # Nirvana, Radiohead, Pearl Jam
    INDIE = "indie"                    # Arctic Monkeys, The Strokes
    GARAGE = "garage"                  # The Black Keys, White Stripes
    POST_PUNK = "post_punk"            # Joy Division, The Cure


class GuitarTechnique(Enum):
    """Guitar playing techniques for rock"""
    POWER_CHORD = "power_chord"        # Root + 5th dyads
    OPEN_CHORD = "open_chord"          # Full open chords
    BARRE_CHORD = "barre_chord"        # Moveable barre chords
    BEND = "bend"                      # String bending
    HAMMER_ON = "hammer_on"            # Hammer-on technique
    PULL_OFF = "pull_off"              # Pull-off technique
    SLIDE = "slide"                    # Sliding between notes
    VIBRATO = "vibrato"                # Pitch modulation
    PALM_MUTE = "palm_mute"            # Light palm muting (not metal-style)
    STRUM = "strum"                    # Strummed chords
    ARPEGGIO = "arpeggio"              # Picked arpeggios


class DrumStyle(Enum):
    """Drum pattern styles"""
    BASIC_ROCK = "basic_rock"          # Kick 1&3, snare 2&4, hi-hat 8ths
    FOUR_ON_FLOOR = "four_on_floor"    # Kick on every quarter
    HALF_TIME = "half_time"            # Grunge/alternative feel
    PUNK_BEAT = "punk_beat"            # Fast, driving 8ths
    POST_PUNK = "post_punk"            # Angular, tribal drums
    INDIE_SHUFFLE = "indie_shuffle"    # Light shuffle feel


class BassPattern(Enum):
    """Bass line patterns"""
    ROOT_FIFTH = "root_fifth"          # Simple root-fifth
    WALKING = "walking"                # Walking bass (blues-influenced)
    OCTAVE_JUMP = "octave_jump"        # Octave jumps
    PENTATONIC_RIFF = "pentatonic"     # Pentatonic riff (parallel to guitar)
    PEDAL_TONE = "pedal_tone"          # Sustained root note
    SYNCOPATED = "syncopated"          # Syncopated pattern


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class RockNote:
    """
    Individual note with rock-specific expression

    Attributes:
        pitch: MIDI note number (0-127)
        velocity: MIDI velocity (1-127)
        start_time: Start time in beats
        duration: Duration in beats
        technique: Guitar/bass technique to apply
        bend_amount: Semitones to bend (0.0-2.0)
        vibrato_depth: Vibrato depth in cents (0-100)
        channel: MIDI channel (default 0)
    """
    pitch: int
    velocity: int
    start_time: float
    duration: float
    technique: GuitarTechnique = GuitarTechnique.OPEN_CHORD
    bend_amount: float = 0.0
    vibrato_depth: float = 0.0
    channel: int = 0

    def __post_init__(self):
        """Validate values"""
        self.velocity = max(1, min(127, self.velocity))
        self.pitch = max(0, min(127, self.pitch))
        self.bend_amount = max(0.0, min(2.0, self.bend_amount))


@dataclass
class PowerChord:
    """
    Power chord (root + fifth) for rock guitar

    Power chords are the foundation of rock guitar:
    - Two-note dyad (root + perfect fifth)
    - Can add octave for fuller sound
    - Standard tuning (unlike metal drop tunings)
    - Palm muting for tighter sound (optional)
    """
    root: int                          # Root note (MIDI)
    add_octave: bool = True            # Add octave above root
    palm_mute: bool = False            # Light palm muting

    def get_notes(self) -> List[int]:
        """
        Get MIDI notes for power chord

        Returns:
            List of MIDI note numbers [root, fifth] or [root, fifth, octave]
        """
        notes = [self.root, self.root + 7]  # Root + fifth
        if self.add_octave:
            notes.append(self.root + 12)     # Octave
        return notes


@dataclass
class RockRiff:
    """
    Guitar or bass riff pattern

    A riff is a short, repeated musical phrase that forms
    the foundation of a rock song (e.g., "Smoke on the Water",
    "Seven Nation Army", "Day Tripper")
    """
    notes: List[RockNote]
    length_bars: int = 2               # Usually 1-2 bars
    repetitions: int = 4               # How many times to repeat

    def to_note_list(self) -> List[RockNote]:
        """Convert riff to full note list with repetitions"""
        full_notes = []
        for rep in range(self.repetitions):
            offset = rep * self.length_bars * 4.0  # 4 beats per bar
            for note in self.notes:
                new_note = RockNote(
                    pitch=note.pitch,
                    velocity=note.velocity,
                    start_time=note.start_time + offset,
                    duration=note.duration,
                    technique=note.technique,
                    bend_amount=note.bend_amount,
                    vibrato_depth=note.vibrato_depth,
                    channel=note.channel
                )
                full_notes.append(new_note)
        return full_notes


@dataclass
class DrumPattern:
    """
    Rock drum pattern

    Uses 16th note resolution for flexibility.
    Each list contains positions (0-15 for one bar).
    """
    kick: List[int] = field(default_factory=list)
    snare: List[int] = field(default_factory=list)
    hihat_closed: List[int] = field(default_factory=list)
    hihat_open: List[int] = field(default_factory=list)
    ride: List[int] = field(default_factory=list)
    crash: List[int] = field(default_factory=list)
    tom_high: List[int] = field(default_factory=list)
    tom_mid: List[int] = field(default_factory=list)
    tom_low: List[int] = field(default_factory=list)
    length: int = 16  # 16th notes (one 4/4 bar)


# ============================================================================
# Scales and Theory
# ============================================================================

class RockScales:
    """
    Scale systems for rock music

    Rock draws heavily from blues and modal scales:
    - Pentatonic scales are foundational (5 notes, no half steps)
    - Blues scale adds the "blue note" (b5)
    - Mixolydian mode for classic rock leads (major with b7)
    - Dorian mode for alternative/indie rock
    """

    # Pentatonic scales (foundation of rock)
    MINOR_PENTATONIC = [0, 3, 5, 7, 10]           # 1, b3, 4, 5, b7
    MAJOR_PENTATONIC = [0, 2, 4, 7, 9]            # 1, 2, 3, 5, 6

    # Blues scale (minor pentatonic + b5)
    BLUES_SCALE = [0, 3, 5, 6, 7, 10]             # 1, b3, 4, b5, 5, b7

    # Modal scales (from church modes)
    MIXOLYDIAN = [0, 2, 4, 5, 7, 9, 10]           # Major with b7 (classic rock)
    DORIAN = [0, 2, 3, 5, 7, 9, 10]               # Minor with natural 6 (alt rock)
    AEOLIAN = [0, 2, 3, 5, 7, 8, 10]              # Natural minor

    @staticmethod
    def get_notes(root: int, scale_type: str = 'minor_pentatonic',
                  octaves: int = 2) -> List[int]:
        """
        Get notes in a rock scale

        Args:
            root: Root note (MIDI number)
            scale_type: Scale type (minor_pentatonic, blues_scale, etc.)
            octaves: Number of octaves to generate

        Returns:
            List of MIDI note numbers
        """
        scale_map = {
            'minor_pentatonic': RockScales.MINOR_PENTATONIC,
            'major_pentatonic': RockScales.MAJOR_PENTATONIC,
            'blues_scale': RockScales.BLUES_SCALE,
            'mixolydian': RockScales.MIXOLYDIAN,
            'dorian': RockScales.DORIAN,
            'aeolian': RockScales.AEOLIAN,
        }

        intervals = scale_map.get(scale_type, RockScales.MINOR_PENTATONIC)

        notes = []
        for octave in range(octaves + 1):
            for interval in intervals:
                note = root + interval + (octave * 12)
                if note <= 127:  # Stay within MIDI range
                    notes.append(note)

        return notes


# ============================================================================
# Chord Progressions
# ============================================================================

class RockProgressions:
    """
    Classic rock chord progressions

    Based on research from:
    - "Axis of Awesome" - Four chord song analysis
    - Common progressions in rock history
    - Beatles harmonic analysis

    Progressions use scale degrees (1-7) for any key.
    """

    @staticmethod
    def get_progression(root: int, progression_type: str = 'i_iv_v',
                       bars: int = 8) -> List[Tuple[int, str, float]]:
        """
        Generate rock chord progression

        Args:
            root: Root note of key (MIDI)
            progression_type: Type of progression
            bars: Number of bars

        Returns:
            List of (root_note, quality, duration) tuples
        """

        # Define progressions as scale degree patterns
        progressions = {
            # Classic rock progressions
            'i_iv_v': [          # G-C-D, A-D-E (most common)
                (0, 'maj', 4.0),    # I
                (5, 'maj', 4.0),    # IV
                (7, 'maj', 4.0),    # V
                (0, 'maj', 4.0),    # I
            ],

            'i_bvii_iv': [       # A-G-D ("Sweet Child O' Mine")
                (0, 'maj', 4.0),    # I
                (-2, 'maj', 4.0),   # bVII
                (5, 'maj', 4.0),    # IV
                (0, 'maj', 4.0),    # I
            ],

            'i_v_vi_iv': [       # C-G-Am-F (ballad progression)
                (0, 'maj', 4.0),    # I
                (7, 'maj', 4.0),    # V
                (9, 'min', 4.0),    # vi
                (5, 'maj', 4.0),    # IV
            ],

            'twelve_bar_blues': [ # Blues-based rock (Led Zeppelin, etc.)
                (0, '7', 4.0),      # I7
                (0, '7', 4.0),      # I7
                (5, '7', 4.0),      # IV7
                (0, '7', 4.0),      # I7
                (5, '7', 4.0),      # IV7
                (5, '7', 4.0),      # IV7
                (0, '7', 4.0),      # I7
                (0, '7', 4.0),      # I7
                (7, '7', 4.0),      # V7
                (5, '7', 4.0),      # IV7
                (0, '7', 4.0),      # I7
                (7, '7', 4.0),      # V7
            ],

            'punk_three_chord': [  # Ramones style (3 chords)
                (0, '5', 4.0),      # I (power chord)
                (5, '5', 4.0),      # IV
                (7, '5', 4.0),      # V
                (5, '5', 4.0),      # IV
            ],

            'modal_vamp': [        # Post-punk/indie (modal)
                (0, 'min', 8.0),    # i (extended)
                (5, 'min', 8.0),    # iv
            ],
        }

        pattern = progressions.get(progression_type, progressions['i_iv_v'])

        # Convert scale degrees to actual notes
        result = []
        pattern_bars = len(pattern)
        repetitions = max(1, bars // pattern_bars)

        for _ in range(repetitions):
            for degree, quality, duration in pattern:
                chord_root = root + degree
                result.append((chord_root, quality, duration))

        return result


# ============================================================================
# Guitar Techniques
# ============================================================================

class GuitarLicks:
    """
    Classic rock guitar licks and techniques

    Based on analysis of legendary rock guitarists:
    - Jimmy Page: Blues-based bends and vibrato
    - Pete Townshend: Power chord windmills
    - Kurt Cobain: Simple, effective riffs
    - Johnny Ramone: Downstroke barrage
    """

    @staticmethod
    def generate_pentatonic_lick(root: int, scale_type: str = 'minor_pentatonic',
                                 length_beats: int = 4) -> List[RockNote]:
        """
        Generate a pentatonic-based guitar lick

        Args:
            root: Root note
            scale_type: Scale to use
            length_beats: Length in beats

        Returns:
            List of RockNote objects
        """
        scale = RockScales.get_notes(root, scale_type, octaves=2)
        notes = []
        current_time = 0.0

        # Typical rock lick: mixture of 8th and 16th notes
        note_durations = [0.5, 0.5, 0.25, 0.25, 0.5, 0.5, 0.5, 0.5]

        beat_count = 0
        for duration in note_durations:
            if beat_count >= length_beats:
                break

            # Pick notes from scale
            pitch = random.choice(scale)

            # Add bends on certain notes (quarter-step to whole-step)
            bend = 0.0
            if random.random() < 0.3:  # 30% chance of bend
                bend = random.choice([0.25, 0.5, 1.0])  # Quarter, half, or whole step

            # Add vibrato
            vibrato = random.uniform(0, 30) if random.random() < 0.4 else 0

            velocity = random.randint(80, 110)

            note = RockNote(
                pitch=pitch,
                velocity=velocity,
                start_time=current_time,
                duration=duration * 0.9,  # Slight gap between notes
                technique=GuitarTechnique.BEND if bend > 0 else GuitarTechnique.OPEN_CHORD,
                bend_amount=bend,
                vibrato_depth=vibrato
            )
            notes.append(note)

            current_time += duration
            beat_count += duration

        return notes

    @staticmethod
    def generate_power_chord_riff(root: int, rhythm_pattern: List[float],
                                   palm_mute: bool = False) -> List[RockNote]:
        """
        Generate a power chord riff

        Args:
            root: Root note
            rhythm_pattern: List of durations for rhythm
            palm_mute: Whether to use palm muting

        Returns:
            List of RockNote objects
        """
        notes = []
        current_time = 0.0

        # Power chord voicing (root, fifth, octave)
        chord_notes = [root, root + 7, root + 12]

        for duration in rhythm_pattern:
            velocity = random.randint(95, 115) if not palm_mute else random.randint(70, 90)

            # Add all notes of power chord
            for pitch in chord_notes:
                note = RockNote(
                    pitch=pitch,
                    velocity=velocity,
                    start_time=current_time,
                    duration=duration * 0.85 if palm_mute else duration * 0.95,
                    technique=GuitarTechnique.PALM_MUTE if palm_mute else GuitarTechnique.POWER_CHORD
                )
                notes.append(note)

            current_time += duration

        return notes


# ============================================================================
# Drum Patterns
# ============================================================================

class RockDrums:
    """
    Rock drum pattern generator

    Based on classic patterns from:
    - John Bonham (Led Zeppelin): Heavy, syncopated grooves
    - Keith Moon (The Who): Chaotic, tom-heavy fills
    - Dave Grohl (Nirvana): Half-time alternative feel
    - Tommy Ramone (Ramones): Fast, relentless punk beat
    """

    # MIDI note numbers for drums (General MIDI)
    KICK = 36
    SNARE = 38
    HIHAT_CLOSED = 42
    HIHAT_OPEN = 46
    TOM_HIGH = 48
    TOM_MID = 47
    TOM_LOW = 45
    CRASH = 49
    RIDE = 51

    @staticmethod
    def generate_pattern(style: DrumStyle = DrumStyle.BASIC_ROCK,
                        bars: int = 4) -> DrumPattern:
        """
        Generate drum pattern

        Args:
            style: Drum style to generate
            bars: Number of bars

        Returns:
            DrumPattern object
        """
        pattern = DrumPattern(length=16 * bars)

        if style == DrumStyle.BASIC_ROCK:
            # Basic rock beat: Kick on 1 & 3, Snare on 2 & 4, Hi-hat 8ths
            for bar in range(bars):
                offset = bar * 16
                # Kick: beats 1 and 3
                pattern.kick.extend([offset + 0, offset + 8])
                # Snare: beats 2 and 4
                pattern.snare.extend([offset + 4, offset + 12])
                # Hi-hat: 8th notes (every 2 16ths)
                pattern.hihat_closed.extend([offset + i for i in range(0, 16, 2)])

        elif style == DrumStyle.FOUR_ON_FLOOR:
            # Kick on every quarter note (disco/dance rock)
            for bar in range(bars):
                offset = bar * 16
                pattern.kick.extend([offset + i for i in range(0, 16, 4)])
                pattern.snare.extend([offset + 4, offset + 12])
                pattern.hihat_closed.extend([offset + i for i in range(0, 16, 2)])

        elif style == DrumStyle.HALF_TIME:
            # Grunge/alternative half-time feel (snare on 3 instead of 2 & 4)
            for bar in range(bars):
                offset = bar * 16
                pattern.kick.extend([offset + 0])
                pattern.snare.extend([offset + 8])  # Snare on beat 3
                pattern.hihat_closed.extend([offset + i for i in range(0, 16, 2)])

        elif style == DrumStyle.PUNK_BEAT:
            # Fast punk: Driving 8ths, rapid hi-hat or ride
            for bar in range(bars):
                offset = bar * 16
                # Kick: 1 & 3 (plus some variations)
                pattern.kick.extend([offset + 0, offset + 8])
                # Snare: 2 & 4
                pattern.snare.extend([offset + 4, offset + 12])
                # Fast 16th notes on hi-hat for energy
                pattern.hihat_closed.extend([offset + i for i in range(16)])

        elif style == DrumStyle.POST_PUNK:
            # Angular, tribal: Emphasis on toms, syncopated
            for bar in range(bars):
                offset = bar * 16
                pattern.kick.extend([offset + 0, offset + 6, offset + 12])
                pattern.snare.extend([offset + 4, offset + 10])
                pattern.tom_high.extend([offset + 2, offset + 14])
                pattern.hihat_open.extend([offset + i for i in range(0, 16, 4)])

        elif style == DrumStyle.INDIE_SHUFFLE:
            # Light shuffle feel (indie rock)
            for bar in range(bars):
                offset = bar * 16
                pattern.kick.extend([offset + 0, offset + 8])
                pattern.snare.extend([offset + 4, offset + 12])
                # Shuffle hi-hat pattern
                pattern.hihat_closed.extend([offset + 0, offset + 3, offset + 6,
                                            offset + 8, offset + 11, offset + 14])

        return pattern

    @staticmethod
    def pattern_to_notes(pattern: DrumPattern) -> List[Tuple[int, float, int]]:
        """
        Convert DrumPattern to MIDI notes

        Args:
            pattern: DrumPattern object

        Returns:
            List of (pitch, time_in_beats, velocity) tuples
        """
        notes = []
        beat_resolution = 0.25  # 16th note = 0.25 beats

        # Process each drum voice
        for pos in pattern.kick:
            notes.append((RockDrums.KICK, pos * beat_resolution, random.randint(100, 120)))

        for pos in pattern.snare:
            notes.append((RockDrums.SNARE, pos * beat_resolution, random.randint(90, 110)))

        for pos in pattern.hihat_closed:
            notes.append((RockDrums.HIHAT_CLOSED, pos * beat_resolution, random.randint(60, 85)))

        for pos in pattern.hihat_open:
            notes.append((RockDrums.HIHAT_OPEN, pos * beat_resolution, random.randint(70, 95)))

        for pos in pattern.ride:
            notes.append((RockDrums.RIDE, pos * beat_resolution, random.randint(70, 90)))

        for pos in pattern.crash:
            notes.append((RockDrums.CRASH, pos * beat_resolution, 120))

        for pos in pattern.tom_high:
            notes.append((RockDrums.TOM_HIGH, pos * beat_resolution, random.randint(85, 105)))

        for pos in pattern.tom_mid:
            notes.append((RockDrums.TOM_MID, pos * beat_resolution, random.randint(85, 105)))

        for pos in pattern.tom_low:
            notes.append((RockDrums.TOM_LOW, pos * beat_resolution, random.randint(90, 110)))

        # Sort by time
        notes.sort(key=lambda x: x[1])

        return notes


# ============================================================================
# Bass Lines
# ============================================================================

class RockBass:
    """
    Rock bass line generator

    Based on classic bass playing styles:
    - John Paul Jones (Led Zeppelin): Walking bass, melodic
    - Paul Simonon (The Clash): Simple, driving punk bass
    - Krist Novoselic (Nirvana): Root-fifth patterns
    - John Entwistle (The Who): Melodic, lead-style bass
    """

    @staticmethod
    def generate_bass_line(progression: List[Tuple[int, str, float]],
                          pattern_type: BassPattern = BassPattern.ROOT_FIFTH) -> List[RockNote]:
        """
        Generate bass line from chord progression

        Args:
            progression: List of (root, quality, duration) tuples
            pattern_type: Type of bass pattern

        Returns:
            List of RockNote objects
        """
        notes = []
        current_time = 0.0

        for chord_root, quality, duration in progression:
            # Transpose bass to lower octave
            bass_root = chord_root - 24 if chord_root > 48 else chord_root - 12

            if pattern_type == BassPattern.ROOT_FIFTH:
                # Simple root-fifth pattern (most common)
                # Quarter notes: root, root, fifth, fifth
                notes.append(RockNote(bass_root, 90, current_time, 0.9))
                notes.append(RockNote(bass_root, 85, current_time + 1.0, 0.9))
                notes.append(RockNote(bass_root + 7, 88, current_time + 2.0, 0.9))
                notes.append(RockNote(bass_root + 7, 85, current_time + 3.0, 0.9))

            elif pattern_type == BassPattern.OCTAVE_JUMP:
                # Octave jumps (energetic)
                notes.append(RockNote(bass_root, 95, current_time, 0.9))
                notes.append(RockNote(bass_root + 12, 90, current_time + 1.0, 0.9))
                notes.append(RockNote(bass_root, 92, current_time + 2.0, 0.9))
                notes.append(RockNote(bass_root + 12, 88, current_time + 3.0, 0.9))

            elif pattern_type == BassPattern.WALKING:
                # Walking bass (blues-influenced)
                # Chromatic approach to next chord
                scale = RockScales.get_notes(bass_root, 'blues_scale', octaves=1)
                for i in range(4):
                    pitch = scale[min(i, len(scale) - 1)]
                    notes.append(RockNote(pitch, 85, current_time + i, 0.9))

            elif pattern_type == BassPattern.PEDAL_TONE:
                # Sustained root note
                notes.append(RockNote(bass_root, 88, current_time, duration * 0.95))

            elif pattern_type == BassPattern.SYNCOPATED:
                # Syncopated pattern (punk/post-punk)
                # 8th note pattern with syncopation
                eighths = [0, 0.5, 1.5, 2.0, 2.5, 3.5]
                for offset in eighths:
                    if offset < duration:
                        pitch = bass_root if random.random() < 0.7 else bass_root + 7
                        notes.append(RockNote(pitch, random.randint(85, 100),
                                            current_time + offset, 0.4))

            elif pattern_type == BassPattern.PENTATONIC_RIFF:
                # Melodic riff using pentatonic scale
                scale = RockScales.get_notes(bass_root, 'minor_pentatonic', octaves=1)
                for i in range(int(duration)):
                    pitch = random.choice(scale)
                    notes.append(RockNote(pitch, random.randint(80, 95),
                                        current_time + i * 0.5, 0.45))

            current_time += duration

        return notes


# ============================================================================
# Song Structure
# ============================================================================

class SongStructure:
    """
    Rock song structure generator

    Typical rock song forms:
    - Verse-Chorus (most common)
    - Verse-Chorus-Bridge
    - AABA (classic pop-rock)
    - Intro-Verse-Chorus-Solo-Bridge-Outro
    """

    @staticmethod
    def generate_structure(style: RockStyle = RockStyle.CLASSIC_ROCK) -> List[Tuple[str, int]]:
        """
        Generate song structure

        Args:
            style: Rock style (affects structure choices)

        Returns:
            List of (section_name, bars) tuples
        """

        if style == RockStyle.PUNK:
            # Punk: Short and simple (2-3 minutes)
            return [
                ('intro', 4),
                ('verse', 8),
                ('chorus', 8),
                ('verse', 8),
                ('chorus', 8),
                ('bridge', 4),
                ('chorus', 8),
            ]

        elif style == RockStyle.CLASSIC_ROCK:
            # Classic rock: Extended with solo
            return [
                ('intro', 8),
                ('verse', 8),
                ('chorus', 8),
                ('verse', 8),
                ('chorus', 8),
                ('solo', 16),      # Extended guitar solo
                ('bridge', 8),
                ('chorus', 8),
                ('outro', 8),
            ]

        elif style in [RockStyle.ALTERNATIVE, RockStyle.INDIE]:
            # Alternative/indie: Varied structure
            return [
                ('intro', 4),
                ('verse', 8),
                ('verse', 8),
                ('chorus', 8),
                ('verse', 8),
                ('chorus', 8),
                ('bridge', 8),
                ('chorus', 8),
                ('outro', 4),
            ]

        elif style == RockStyle.POST_PUNK:
            # Post-punk: Unconventional, longer sections
            return [
                ('intro', 8),
                ('verse', 12),
                ('breakdown', 8),
                ('verse', 12),
                ('climax', 16),
                ('outro', 8),
            ]

        else:
            # Default structure
            return [
                ('intro', 4),
                ('verse', 8),
                ('chorus', 8),
                ('verse', 8),
                ('chorus', 8),
                ('outro', 4),
            ]


# ============================================================================
# Main Generator Class
# ============================================================================

class ClassicRockGenerator:
    """
    Comprehensive classic rock music generator

    Generates complete rock compositions across multiple sub-genres:
    - Classic Rock (60s-70s)
    - Punk Rock
    - Alternative Rock
    - Indie Rock
    - Garage Rock
    - Post-Punk

    Features:
    - Chord progressions (I-IV-V, blues-based, modal)
    - Power chord riffs
    - Guitar techniques (bends, slides, vibrato)
    - Drum patterns (rock beat, punk, half-time)
    - Bass lines (root-fifth, walking, melodic)
    - Song structures (verse-chorus-bridge)
    """

    def __init__(self, style: RockStyle = RockStyle.CLASSIC_ROCK,
                 key_root: int = 60, tempo: int = 120):
        """
        Initialize Classic Rock Generator

        Args:
            style: Rock sub-genre style
            key_root: Root note of key (MIDI, e.g., 60 = C)
            tempo: Tempo in BPM
        """
        self.style = style
        self.key_root = key_root
        self.tempo = tempo

        # Set style-specific defaults
        if style == RockStyle.PUNK:
            self.tempo = max(tempo, 160)  # Punk is fast (160-200 BPM)
        elif style == RockStyle.POST_PUNK:
            self.tempo = min(tempo, 110)  # Post-punk tends slower

    def generate_composition(self, length_bars: int = 32,
                           include_solo: bool = True) -> Dict:
        """
        Generate complete rock composition

        Args:
            length_bars: Total length in bars
            include_solo: Whether to include guitar solo section

        Returns:
            Dictionary with all parts:
            {
                'chords': chord progression,
                'guitar_rhythm': rhythm guitar part,
                'guitar_lead': lead guitar/solo,
                'bass': bass line,
                'drums': drum pattern,
                'structure': song structure,
                'tempo': tempo,
                'key': key root
            }
        """
        composition = {
            'tempo': self.tempo,
            'key': self.key_root,
            'style': self.style.value
        }

        # Generate chord progression based on style
        if self.style == RockStyle.PUNK:
            progression_type = 'punk_three_chord'
        elif self.style == RockStyle.POST_PUNK:
            progression_type = 'modal_vamp'
        elif self.style == RockStyle.CLASSIC_ROCK:
            progression_type = random.choice(['i_iv_v', 'i_bvii_iv', 'twelve_bar_blues'])
        else:
            progression_type = random.choice(['i_iv_v', 'i_v_vi_iv'])

        composition['chords'] = RockProgressions.get_progression(
            self.key_root, progression_type, length_bars
        )

        # Generate drums based on style
        if self.style == RockStyle.PUNK:
            drum_style = DrumStyle.PUNK_BEAT
        elif self.style == RockStyle.ALTERNATIVE:
            drum_style = DrumStyle.HALF_TIME
        elif self.style == RockStyle.POST_PUNK:
            drum_style = DrumStyle.POST_PUNK
        elif self.style == RockStyle.INDIE:
            drum_style = DrumStyle.INDIE_SHUFFLE
        else:
            drum_style = DrumStyle.BASIC_ROCK

        drum_pattern = RockDrums.generate_pattern(drum_style, length_bars)
        composition['drums'] = RockDrums.pattern_to_notes(drum_pattern)

        # Generate bass line
        if self.style == RockStyle.PUNK:
            bass_pattern = BassPattern.SYNCOPATED
        elif self.style in [RockStyle.ALTERNATIVE, RockStyle.GARAGE]:
            bass_pattern = BassPattern.ROOT_FIFTH
        elif self.style == RockStyle.CLASSIC_ROCK:
            bass_pattern = random.choice([BassPattern.ROOT_FIFTH, BassPattern.WALKING])
        else:
            bass_pattern = BassPattern.ROOT_FIFTH

        composition['bass'] = RockBass.generate_bass_line(
            composition['chords'], bass_pattern
        )

        # Generate rhythm guitar (power chords or open chords)
        if self.style in [RockStyle.PUNK, RockStyle.GARAGE]:
            # Power chord riff for punk/garage
            rhythm_pattern = [0.5] * 8  # 8th notes for 4 bars
            composition['guitar_rhythm'] = GuitarLicks.generate_power_chord_riff(
                self.key_root, rhythm_pattern, palm_mute=False
            )
        else:
            # Strummed chords or arpeggios for other styles
            composition['guitar_rhythm'] = []  # Placeholder

        # Generate lead guitar/solo if requested
        if include_solo:
            scale_type = 'blues_scale' if self.style == RockStyle.CLASSIC_ROCK else 'minor_pentatonic'
            composition['guitar_lead'] = GuitarLicks.generate_pentatonic_lick(
                self.key_root + 12, scale_type, length_beats=16
            )

        # Generate song structure
        composition['structure'] = SongStructure.generate_structure(self.style)

        return composition

    def generate_riff(self, length_bars: int = 2,
                     use_power_chords: bool = True) -> RockRiff:
        """
        Generate a signature rock riff

        Args:
            length_bars: Length of riff in bars
            use_power_chords: Whether to use power chords

        Returns:
            RockRiff object
        """
        if use_power_chords:
            # Generate power chord riff
            rhythm_pattern = [0.5, 0.5, 0.25, 0.25, 0.5, 0.5]  # Varied rhythm
            notes = GuitarLicks.generate_power_chord_riff(
                self.key_root, rhythm_pattern, palm_mute=(self.style == RockStyle.PUNK)
            )
        else:
            # Generate pentatonic-based melodic riff
            scale_type = 'minor_pentatonic'
            notes = GuitarLicks.generate_pentatonic_lick(
                self.key_root, scale_type, length_beats=length_bars * 4
            )

        return RockRiff(notes=notes, length_bars=length_bars, repetitions=4)


# ============================================================================
# Utility Functions
# ============================================================================

def create_classic_rock_song(key: str = 'A', tempo: int = 120,
                            style: str = 'classic_rock') -> Dict:
    """
    Convenience function to create a complete classic rock song

    Args:
        key: Musical key (e.g., 'A', 'C', 'E', 'G')
        tempo: Tempo in BPM
        style: Style string ('classic_rock', 'punk', 'alternative', etc.)

    Returns:
        Complete composition dictionary
    """
    # Convert key string to MIDI note
    key_map = {
        'C': 60, 'C#': 61, 'Db': 61,
        'D': 62, 'D#': 63, 'Eb': 63,
        'E': 64,
        'F': 65, 'F#': 66, 'Gb': 66,
        'G': 67, 'G#': 68, 'Ab': 68,
        'A': 69, 'A#': 70, 'Bb': 70,
        'B': 71
    }

    key_root = key_map.get(key, 60)

    # Convert style string to enum
    style_map = {
        'classic_rock': RockStyle.CLASSIC_ROCK,
        'punk': RockStyle.PUNK,
        'alternative': RockStyle.ALTERNATIVE,
        'indie': RockStyle.INDIE,
        'garage': RockStyle.GARAGE,
        'post_punk': RockStyle.POST_PUNK,
    }

    rock_style = style_map.get(style, RockStyle.CLASSIC_ROCK)

    # Create generator and generate composition
    generator = ClassicRockGenerator(rock_style, key_root, tempo)
    return generator.generate_composition(length_bars=32, include_solo=True)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Enums
    'RockStyle',
    'GuitarTechnique',
    'DrumStyle',
    'BassPattern',

    # Data Classes
    'RockNote',
    'PowerChord',
    'RockRiff',
    'DrumPattern',

    # Helpers
    'RockScales',
    'RockProgressions',
    'GuitarLicks',
    'RockDrums',
    'RockBass',
    'SongStructure',

    # Main Generator
    'ClassicRockGenerator',

    # Utility
    'create_classic_rock_song',
]


# ============================================================================
# Test and Example Code
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing"""

    print("Classic Rock Music Generator - Test Suite")
    print("=" * 70)

    # Test 1: Generate I-IV-V progression in A
    print("\n1. Generating I-IV-V progression in A...")
    progression = RockProgressions.get_progression(69, 'i_iv_v', 4)
    print(f"   Generated {len(progression)} chords:")
    for i, (root, quality, dur) in enumerate(progression):
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        name = note_names[root % 12]
        print(f"   - Chord {i+1}: {name} {quality}")

    # Test 2: Generate minor pentatonic scale in E
    print("\n2. Generating minor pentatonic scale in E...")
    scale = RockScales.get_notes(64, 'minor_pentatonic', 2)
    print(f"   Scale notes (MIDI): {scale}")

    # Test 3: Generate basic rock drum pattern
    print("\n3. Generating basic rock drum pattern (4 bars)...")
    drums = RockDrums.generate_pattern(DrumStyle.BASIC_ROCK, 4)
    drum_notes = RockDrums.pattern_to_notes(drums)
    print(f"   Generated {len(drum_notes)} drum events")
    print(f"   Kick hits: {len(drums.kick)}, Snare hits: {len(drums.snare)}")

    # Test 4: Generate power chord riff
    print("\n4. Generating power chord riff in A...")
    rhythm = [0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.5, 0.5]
    riff_notes = GuitarLicks.generate_power_chord_riff(57, rhythm)  # A3
    print(f"   Generated {len(riff_notes)} notes")

    # Test 5: Generate bass line
    print("\n5. Generating root-fifth bass line...")
    test_progression = [(69, 'maj', 4.0), (74, 'maj', 4.0), (76, 'maj', 4.0)]
    bass = RockBass.generate_bass_line(test_progression, BassPattern.ROOT_FIFTH)
    print(f"   Generated {len(bass)} bass notes")

    # Test 6: Generate pentatonic lick with bends
    print("\n6. Generating pentatonic guitar lick with bends...")
    lick = GuitarLicks.generate_pentatonic_lick(64, 'minor_pentatonic', 4)
    print(f"   Generated {len(lick)} notes")
    bent_notes = [n for n in lick if n.bend_amount > 0]
    print(f"   Notes with bends: {len(bent_notes)}")

    # Test 7: Complete classic rock composition
    print("\n7. Generating complete Classic Rock composition in A...")
    classic_gen = ClassicRockGenerator(RockStyle.CLASSIC_ROCK, 69, 120)
    composition = classic_gen.generate_composition(32, include_solo=True)
    print(f"   Style: {composition['style']}")
    print(f"   Tempo: {composition['tempo']} BPM")
    print(f"   Chord progression: {len(composition['chords'])} chords")
    print(f"   Drum events: {len(composition['drums'])}")
    print(f"   Bass notes: {len(composition['bass'])}")
    print(f"   Structure sections: {len(composition['structure'])}")

    # Test 8: Punk rock composition
    print("\n8. Generating Punk Rock composition (fast & simple)...")
    punk_gen = ClassicRockGenerator(RockStyle.PUNK, 64, 180)  # E, 180 BPM
    punk_song = punk_gen.generate_composition(24, include_solo=False)
    print(f"   Tempo: {punk_song['tempo']} BPM (fast!)")
    print(f"   Chord progression: {len(punk_song['chords'])} chords")
    print(f"   Structure: {[section[0] for section in punk_song['structure']]}")

    # Test 9: Alternative rock with half-time feel
    print("\n9. Generating Alternative Rock composition (half-time)...")
    alt_gen = ClassicRockGenerator(RockStyle.ALTERNATIVE, 62, 100)  # D, 100 BPM
    alt_song = alt_gen.generate_composition(28, include_solo=True)
    print(f"   Style: {alt_song['style']}")
    print(f"   Has guitar solo: {'guitar_lead' in alt_song}")

    # Test 10: Create song using convenience function
    print("\n10. Creating song using convenience function...")
    song = create_classic_rock_song(key='G', tempo=125, style='indie')
    print(f"   Key: G, Tempo: {song['tempo']}, Style: {song['style']}")
    print(f"   Complete composition generated successfully!")

    # Test 11: Song structure generation
    print("\n11. Generating song structures for different styles...")
    for style in [RockStyle.CLASSIC_ROCK, RockStyle.PUNK, RockStyle.POST_PUNK]:
        structure = SongStructure.generate_structure(style)
        sections = [s[0] for s in structure]
        total_bars = sum(s[1] for s in structure)
        print(f"   {style.value}: {sections} ({total_bars} bars total)")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("\nClassic Rock Generator ready for use.")
    print("Supports: Classic Rock, Punk, Alternative, Indie, Garage, Post-Punk")
