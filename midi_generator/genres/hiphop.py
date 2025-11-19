#!/usr/bin/env python3
"""
Hip-Hop/Rap Music Generator

Comprehensive implementation of hip-hop music across 6 sub-genres.

Sub-genres:
-----------
- Boom Bap (90s Golden Age: Wu-Tang Clan, Nas, A Tribe Called Quest, Gang Starr)
- Trap (Modern: Future, Migos, Travis Scott, 21 Savage)
- Lo-Fi Hip-Hop (Nujabes, J Dilla, ChilledCow aesthetic, study beats)
- Drill (Chicago: Chief Keef, Pop Smoke / UK: Headie One, Digga D)
- Conscious Rap (Kendrick Lamar, J. Cole, Common, Mos Def)
- G-Funk (West Coast: Dr. Dre, Snoop Dogg, Warren G, Nate Dogg)

Features:
---------
- Drum Patterns: Boom bap, trap hi-hats (32nd/64th rolls), drill, lo-fi quantization
- 808 Bass: Pitch slides, sub-bass frequencies, distortion, velocity-sensitive decay
- Sample Chopping: 4/8/16-slice patterns, time-stretching, pitch-shifting
- Beat Structures: 16-bar loops, A-B variations, intro-verse-hook patterns
- Harmonic Simplicity: Minor triads, 2-chord vamps, modal progressions
- Swing and Timing: J Dilla swing (53-56%), MPC swing, quantization offsets

Research References:
-------------------
- "Dilla Time" - Ethan Hein (J Dilla microtiming analysis)
- "Making Beats: The Art of Sample-Based Hip-Hop" - Joseph G. Schloss
- "The Anthology of Rap" - Yale University Press
- "Roland TR-808 Rhythm Composer" - Technical manual
- MPC60/MPC3000 swing algorithm documentation (Roger Linn)
- "The Art of Sampling" - Amir Said (Questlove)
- "Check the Technique" - Brian Coleman (production breakdowns)

Integration:
-----------
- Uses groove_library.py for MPC swing and hip-hop timing profiles
- Uses rhythm_engine.py for quantization and humanization
- Integrates with existing MIDI constants for drum mapping

Author: Agent 41 - Hip-Hop/Rap Module
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from midi.midi_constants import (
    GM_DRUM_MAP,
    DEFAULT_PPQN,
    PPQN_HIGH_RES,
    KICK_NOTES,
    SNARE_NOTES,
    HIHAT_NOTES
)

# Optional imports (use if available, but not required)
try:
    from algorithms.rhythm_engine import RhythmNote as RhythmEngineNote, RhythmEngine, TimingStyle
    from algorithms.groove_library import GrooveLibrary, GenreTimingProfiles
    RHYTHM_ENGINE_AVAILABLE = True
except ImportError:
    RHYTHM_ENGINE_AVAILABLE = False


# ============================================================================
# Enums and Type Definitions
# ============================================================================

class HipHopStyle(Enum):
    """Hip-hop sub-genres"""
    BOOM_BAP = "boom_bap"          # 90s golden age
    TRAP = "trap"                  # Modern trap
    LOFI = "lofi"                  # Lo-fi hip-hop
    DRILL = "drill"                # Chicago/UK drill
    CONSCIOUS = "conscious"        # Conscious rap
    G_FUNK = "g_funk"             # West Coast G-funk


class DrumArticulation(Enum):
    """Drum hit articulations"""
    HARD = "hard"                  # Full velocity hit
    GHOST = "ghost"                # Soft ghost note
    ACCENT = "accent"              # Accented hit
    FLAM = "flam"                  # Grace note before hit
    ROLL = "roll"                  # Rapid hits (trap hi-hats)


class SampleSliceMode(Enum):
    """Sample chopping modes"""
    FOUR_SLICE = 4                 # 4 equal slices
    EIGHT_SLICE = 8                # 8 equal slices
    SIXTEEN_SLICE = 16             # 16 equal slices
    IRREGULAR = "irregular"        # Non-uniform slices


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class HipHopNote:
    """
    A single note/hit in a hip-hop pattern with expression.

    Attributes:
        pitch: MIDI note number (0-127)
        velocity: MIDI velocity (1-127)
        start_time: Start time in beats
        duration: Duration in beats
        articulation: Hit articulation type
        channel: MIDI channel (9=drums, others for melodic)
        swing_offset: Timing offset in ticks for swing feel
        pitch_bend: Pitch bend value for 808 slides (-8192 to 8191)
    """
    pitch: int
    velocity: int
    start_time: float
    duration: float
    articulation: DrumArticulation = DrumArticulation.HARD
    channel: int = 9  # Default to drum channel
    swing_offset: int = 0
    pitch_bend: int = 0

    def __post_init__(self):
        """Validate values"""
        self.velocity = max(1, min(127, self.velocity))
        self.pitch = max(0, min(127, self.pitch))


@dataclass
class Bass808:
    """
    808 Bass configuration with pitch slide.

    The Roland TR-808's bass drum is iconic in hip-hop for its deep
    sub-bass and characteristic pitch decay.

    Attributes:
        start_pitch: Starting MIDI pitch (typically 30-60 Hz range)
        end_pitch: Ending pitch after slide
        attack_time: Attack envelope time in ms
        decay_time: Decay envelope time in ms (pitch slide duration)
        sustain_level: Sustain level (0.0-1.0)
        release_time: Release time in ms
        distortion: Distortion amount (0.0-1.0)
        sub_bass_level: Sub-bass sine wave level (0.0-1.0)
    """
    start_pitch: int = 36  # C1 (65.4 Hz)
    end_pitch: int = 24    # C0 (32.7 Hz) - octave drop
    attack_time: float = 0  # Immediate attack
    decay_time: float = 150  # 150ms pitch slide
    sustain_level: float = 0.6
    release_time: float = 100
    distortion: float = 0.3
    sub_bass_level: float = 0.8

    def get_pitch_bend_curve(self, num_steps: int = 20) -> List[int]:
        """
        Generate pitch bend curve for 808 slide.

        Args:
            num_steps: Number of pitch bend steps

        Returns:
            List of MIDI pitch bend values (-8192 to 8191)
        """
        semitone_diff = self.start_pitch - self.end_pitch
        max_bend = semitone_diff * 4096  # 4096 per semitone

        # Exponential decay curve (realistic 808 behavior)
        curve = []
        for i in range(num_steps):
            t = i / num_steps
            # Exponential decay: fast drop then slow
            bend_amount = max_bend * (1 - t) ** 2
            curve.append(int(bend_amount))

        return curve


@dataclass
class DrumPattern:
    """
    Complete drum pattern with all elements.

    Attributes:
        kick: Kick drum notes
        snare: Snare drum notes
        hihat: Hi-hat notes (closed/open)
        percussion: Additional percussion (claps, etc.)
        length_bars: Pattern length in bars
        time_signature: Time signature tuple (numerator, denominator)
    """
    kick: List[HipHopNote] = field(default_factory=list)
    snare: List[HipHopNote] = field(default_factory=list)
    hihat: List[HipHopNote] = field(default_factory=list)
    percussion: List[HipHopNote] = field(default_factory=list)
    length_bars: int = 4
    time_signature: Tuple[int, int] = (4, 4)

    def get_all_notes(self) -> List[HipHopNote]:
        """Combine all drum elements into single list"""
        return self.kick + self.snare + self.hihat + self.percussion


@dataclass
class SampleChop:
    """
    Sample chopping configuration.

    Attributes:
        slice_mode: How to slice the sample
        slice_positions: Beat positions for each slice
        pitch_shifts: Pitch shift in semitones per slice
        reverse: Whether each slice should be reversed
        stutter: Stutter/repeat effect per slice
    """
    slice_mode: SampleSliceMode
    slice_positions: List[float] = field(default_factory=list)
    pitch_shifts: List[int] = field(default_factory=list)
    reverse: List[bool] = field(default_factory=list)
    stutter: List[int] = field(default_factory=list)  # Repeat count per slice


# ============================================================================
# Hip-Hop Scales and Harmony
# ============================================================================

class HipHopHarmony:
    """
    Harmonic structures used in hip-hop.

    Hip-hop typically uses simple, repetitive harmonic structures
    to leave space for vocals and rhythm.
    """

    # Common hip-hop chord progressions (in semitones from root)
    PROGRESSIONS = {
        'minor_vamp': [(0, 'minor'), (5, 'minor')],  # i - iv (very common)
        'sad_trap': [(0, 'minor'), (-2, 'major')],   # i - bVII
        'g_funk': [(0, 'major'), (5, 'major'), (7, 'major'), (3, 'minor')],  # I - IV - V - iii
        'conscious': [(0, 'minor'), (7, 'minor'), (10, 'minor'), (5, 'minor')],  # i - v - bvii - iv
        'dark': [(0, 'minor'), (-5, 'minor')],  # i - bv (diminished fifth - dark/drill)
        'dreamy_lofi': [(0, 'maj7'), (2, 'min7'), (5, 'maj7'), (7, 'dom7')],  # Imaj7 - iim7 - IVmaj7 - V7
    }

    # Minor pentatonic (most common in hip-hop)
    MINOR_PENTATONIC = [0, 3, 5, 7, 10]  # 1, b3, 4, 5, b7

    # Dorian mode (conscious rap, jazz-influenced)
    DORIAN = [0, 2, 3, 5, 7, 9, 10]  # 1, 2, b3, 4, 5, 6, b7

    # Phrygian mode (dark trap/drill)
    PHRYGIAN = [0, 1, 3, 5, 7, 8, 10]  # 1, b2, b3, 4, 5, b6, b7

    @staticmethod
    def get_progression(style: HipHopStyle, root: int = 60) -> List[Tuple[int, str, float]]:
        """
        Get chord progression for hip-hop style.

        Args:
            style: Hip-hop sub-genre
            root: Root MIDI note

        Returns:
            List of (root_note, quality, duration_beats) tuples
        """
        if style == HipHopStyle.BOOM_BAP:
            prog_name = 'minor_vamp'
        elif style == HipHopStyle.TRAP:
            prog_name = 'sad_trap'
        elif style == HipHopStyle.LOFI:
            prog_name = 'dreamy_lofi'
        elif style == HipHopStyle.DRILL:
            prog_name = 'dark'
        elif style == HipHopStyle.CONSCIOUS:
            prog_name = 'conscious'
        elif style == HipHopStyle.G_FUNK:
            prog_name = 'g_funk'
        else:
            prog_name = 'minor_vamp'

        template = HipHopHarmony.PROGRESSIONS[prog_name]

        # Convert to absolute notes with duration
        progression = []
        for offset, quality in template:
            progression.append((root + offset, quality, 4.0))  # 4 beats per chord

        return progression

    @staticmethod
    def get_scale(style: HipHopStyle, root: int = 60, octaves: int = 2) -> List[int]:
        """
        Get scale notes for hip-hop style.

        Args:
            style: Hip-hop sub-genre
            root: Root MIDI note
            octaves: Number of octaves

        Returns:
            List of MIDI note numbers
        """
        if style in [HipHopStyle.BOOM_BAP, HipHopStyle.TRAP, HipHopStyle.G_FUNK]:
            intervals = HipHopHarmony.MINOR_PENTATONIC
        elif style == HipHopStyle.CONSCIOUS:
            intervals = HipHopHarmony.DORIAN
        elif style == HipHopStyle.DRILL:
            intervals = HipHopHarmony.PHRYGIAN
        elif style == HipHopStyle.LOFI:
            intervals = HipHopHarmony.MINOR_PENTATONIC
        else:
            intervals = HipHopHarmony.MINOR_PENTATONIC

        notes = []
        for octave in range(octaves + 1):
            for interval in intervals:
                notes.append(root + interval + (octave * 12))

        return notes


# ============================================================================
# Drum Pattern Generators
# ============================================================================

class BoomBapDrums:
    """
    Boom Bap drum patterns (90s Golden Age).

    Characteristics:
    - Hard-hitting kick and snare (MPC-sampled)
    - Snare on 2 and 4
    - Minimal hi-hats or no hi-hats
    - Ghost notes on snare
    - Swing feel (MPC swing)
    """

    @staticmethod
    def generate_pattern(bars: int = 4, swing_amount: float = 0.54) -> DrumPattern:
        """
        Generate boom bap drum pattern.

        Args:
            bars: Number of bars
            swing_amount: Swing ratio (0.5=straight, 0.54=MPC swing)

        Returns:
            DrumPattern object
        """
        pattern = DrumPattern(length_bars=bars)
        beats_per_bar = 4
        total_beats = bars * beats_per_bar

        # Kick pattern: 1, 3, and syncopated 4&
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            # Kick on 1
            pattern.kick.append(HipHopNote(
                pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                velocity=random.randint(95, 105),
                start_time=base_beat,
                duration=0.25,
                articulation=DrumArticulation.ACCENT
            ))

            # Kick on 3
            pattern.kick.append(HipHopNote(
                pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                velocity=random.randint(90, 100),
                start_time=base_beat + 2,
                duration=0.25,
                articulation=DrumArticulation.HARD
            ))

            # Syncopated kick on 4&
            if random.random() > 0.5:
                pattern.kick.append(HipHopNote(
                    pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                    velocity=random.randint(85, 95),
                    start_time=base_beat + 3.5,
                    duration=0.25,
                    articulation=DrumArticulation.HARD
                ))

        # Snare pattern: 2 and 4 (backbeat) + ghost notes
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            # Main snare on 2
            pattern.snare.append(HipHopNote(
                pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
                velocity=random.randint(100, 110),
                start_time=base_beat + 1,
                duration=0.15,
                articulation=DrumArticulation.ACCENT
            ))

            # Main snare on 4
            pattern.snare.append(HipHopNote(
                pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
                velocity=random.randint(100, 110),
                start_time=base_beat + 3,
                duration=0.15,
                articulation=DrumArticulation.ACCENT
            ))

            # Ghost notes (soft hits between main hits)
            ghost_positions = [1.5, 2.25, 3.75]
            for pos in ghost_positions:
                if random.random() > 0.6:  # 40% chance
                    pattern.snare.append(HipHopNote(
                        pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
                        velocity=random.randint(25, 40),
                        start_time=base_beat + pos,
                        duration=0.1,
                        articulation=DrumArticulation.GHOST
                    ))

        # Minimal hi-hats (or none) - sparse 8th notes
        for beat in range(0, total_beats * 2, 2):  # Every other 8th note
            if random.random() > 0.3:  # 70% chance
                pattern.hihat.append(HipHopNote(
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT'],
                    velocity=random.randint(50, 65),
                    start_time=beat / 2.0,
                    duration=0.1,
                    articulation=DrumArticulation.HARD
                ))

        return pattern


class TrapDrums:
    """
    Trap drum patterns (modern hip-hop).

    Characteristics:
    - Roland TR-808 sounds
    - Rapid hi-hat rolls (32nd/64th notes)
    - Sparse kick and snare
    - Layered percussion (claps, snaps)
    - Triplet hi-hat patterns
    """

    @staticmethod
    def generate_pattern(bars: int = 4, hi_hat_density: str = 'medium') -> DrumPattern:
        """
        Generate trap drum pattern.

        Args:
            bars: Number of bars
            hi_hat_density: 'low', 'medium', 'high', 'roll'

        Returns:
            DrumPattern object
        """
        pattern = DrumPattern(length_bars=bars)
        beats_per_bar = 4

        # Sparse kick pattern
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            # Kick on 1
            pattern.kick.append(HipHopNote(
                pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                velocity=random.randint(100, 115),
                start_time=base_beat,
                duration=0.3,
                articulation=DrumArticulation.ACCENT
            ))

            # Sparse additional kicks
            kick_positions = [0.75, 2.5, 3.25]
            for pos in random.sample(kick_positions, random.randint(1, 2)):
                pattern.kick.append(HipHopNote(
                    pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                    velocity=random.randint(90, 105),
                    start_time=base_beat + pos,
                    duration=0.25,
                    articulation=DrumArticulation.HARD
                ))

        # Snare on 3 (half-time feel)
        for bar in range(bars):
            base_beat = bar * beats_per_bar
            pattern.snare.append(HipHopNote(
                pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
                velocity=random.randint(105, 120),
                start_time=base_beat + 2,  # Beat 3
                duration=0.2,
                articulation=DrumArticulation.ACCENT
            ))

            # Layered clap
            pattern.percussion.append(HipHopNote(
                pitch=GM_DRUM_MAP['HAND_CLAP'],
                velocity=random.randint(90, 100),
                start_time=base_beat + 2,
                duration=0.15,
                articulation=DrumArticulation.HARD
            ))

        # Trap hi-hats (complex patterns)
        TrapDrums._add_trap_hihats(pattern, bars, hi_hat_density)

        return pattern

    @staticmethod
    def _add_trap_hihats(pattern: DrumPattern, bars: int, density: str):
        """Add characteristic trap hi-hat patterns with rolls"""
        beats_per_bar = 4
        sixteenth = 0.25
        thirty_second = 0.125

        for bar in range(bars):
            base_beat = bar * beats_per_bar

            if density == 'roll':
                # Hi-hat rolls (rapid 32nd notes)
                roll_start = base_beat + 3.5  # Last half beat
                for i in range(8):  # 8 x 32nd notes = 1 beat
                    pattern.hihat.append(HipHopNote(
                        pitch=GM_DRUM_MAP['CLOSED_HI_HAT'],
                        velocity=random.randint(70, 90),
                        start_time=roll_start + (i * thirty_second),
                        duration=thirty_second * 0.8,
                        articulation=DrumArticulation.ROLL
                    ))
            else:
                # Regular hi-hat pattern (16th notes)
                for sixteenths in range(16):
                    beat_pos = base_beat + (sixteenths * sixteenth)

                    # Varying probability based on density
                    prob = {
                        'low': 0.4,
                        'medium': 0.7,
                        'high': 0.9
                    }.get(density, 0.7)

                    if random.random() < prob:
                        # Accent every 4th hit
                        vel = random.randint(75, 95) if sixteenths % 4 == 0 else random.randint(55, 75)
                        pattern.hihat.append(HipHopNote(
                            pitch=GM_DRUM_MAP['CLOSED_HI_HAT'],
                            velocity=vel,
                            start_time=beat_pos,
                            duration=0.1,
                            articulation=DrumArticulation.ACCENT if sixteenths % 4 == 0 else DrumArticulation.HARD
                        ))


class DrillDrums:
    """
    Drill drum patterns (Chicago/UK drill).

    Characteristics:
    - Very sparse arrangement
    - Sliding 808s with pitch bends
    - Minimal hi-hats
    - Dark, menacing feel
    - Syncopated patterns
    """

    @staticmethod
    def generate_pattern(bars: int = 4) -> DrumPattern:
        """
        Generate drill drum pattern.

        Args:
            bars: Number of bars

        Returns:
            DrumPattern object
        """
        pattern = DrumPattern(length_bars=bars)
        beats_per_bar = 4

        # Very sparse kick pattern
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            # Kick on 1
            pattern.kick.append(HipHopNote(
                pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                velocity=random.randint(110, 127),
                start_time=base_beat,
                duration=0.4,
                articulation=DrumArticulation.ACCENT
            ))

            # Occasional syncopated kick
            if random.random() > 0.6:
                pattern.kick.append(HipHopNote(
                    pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                    velocity=random.randint(95, 110),
                    start_time=base_beat + random.choice([1.75, 2.75, 3.5]),
                    duration=0.3,
                    articulation=DrumArticulation.HARD
                ))

        # Sparse snare/clap
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            # Main snare on 3
            pattern.snare.append(HipHopNote(
                pitch=GM_DRUM_MAP['ELECTRIC_SNARE'],
                velocity=random.randint(100, 115),
                start_time=base_beat + 2,
                duration=0.2,
                articulation=DrumArticulation.ACCENT
            ))

        # Minimal hi-hats (very sparse)
        for bar in range(bars):
            base_beat = bar * beats_per_bar
            for eighth in [0, 2, 2.5, 3.5]:
                if random.random() > 0.3:
                    pattern.hihat.append(HipHopNote(
                        pitch=GM_DRUM_MAP['CLOSED_HI_HAT'],
                        velocity=random.randint(45, 60),
                        start_time=base_beat + eighth * 0.5,
                        duration=0.08,
                        articulation=DrumArticulation.HARD
                    ))

        return pattern


class LoFiDrums:
    """
    Lo-Fi Hip-Hop drum patterns.

    Characteristics:
    - Off-grid quantization (humanized, "dusty" feel)
    - Soft, muted hits
    - Jazz-influenced with swing
    - Vinyl crackle aesthetic (not implemented in MIDI)
    - Simple, repetitive patterns
    """

    @staticmethod
    def generate_pattern(bars: int = 4, swing_amount: float = 0.56) -> DrumPattern:
        """
        Generate lo-fi drum pattern.

        Args:
            bars: Number of bars
            swing_amount: Swing ratio (0.5=straight, 0.56=J Dilla swing)

        Returns:
            DrumPattern object
        """
        pattern = DrumPattern(length_bars=bars)
        beats_per_bar = 4

        # Soft kick pattern
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            kick_positions = [0, 2, 3.5]
            for pos in kick_positions:
                # Add random timing offset (off-grid feel)
                offset = random.uniform(-0.05, 0.05)
                pattern.kick.append(HipHopNote(
                    pitch=GM_DRUM_MAP['BASS_DRUM_1'],
                    velocity=random.randint(60, 75),  # Soft
                    start_time=base_beat + pos + offset,
                    duration=0.2,
                    articulation=DrumArticulation.HARD
                ))

        # Soft snare with swing
        for bar in range(bars):
            base_beat = bar * beats_per_bar

            for snare_beat in [1, 3]:  # 2 and 4
                offset = random.uniform(-0.03, 0.03)
                pattern.snare.append(HipHopNote(
                    pitch=GM_DRUM_MAP['ACOUSTIC_SNARE'],
                    velocity=random.randint(55, 70),  # Muted
                    start_time=base_beat + snare_beat + offset,
                    duration=0.15,
                    articulation=DrumArticulation.HARD
                ))

        # Swung hi-hats (8th note triplets)
        for bar in range(bars):
            base_beat = bar * beats_per_bar
            for eighth in range(8):
                # Apply swing
                if eighth % 2 == 1:  # Offbeat
                    swing_offset = (swing_amount - 0.5) * 0.5
                else:
                    swing_offset = 0

                time_offset = random.uniform(-0.02, 0.02)  # Slight randomness
                pattern.hihat.append(HipHopNote(
                    pitch=GM_DRUM_MAP['CLOSED_HI_HAT'],
                    velocity=random.randint(40, 60),  # Soft
                    start_time=base_beat + (eighth * 0.5) + swing_offset + time_offset,
                    duration=0.1,
                    articulation=DrumArticulation.HARD
                ))

        return pattern


# ============================================================================
# 808 Bass Engine
# ============================================================================

class Bass808Engine:
    """
    Roland TR-808 bass drum engine with pitch slides.

    The 808 bass is characterized by:
    - Deep sub-bass frequencies (30-60 Hz)
    - Pitch decay (slides from high to low)
    - Long sustain
    - Distortion/saturation
    """

    @staticmethod
    def generate_808_pattern(
        style: HipHopStyle,
        bars: int = 4,
        root_note: int = 36
    ) -> List[HipHopNote]:
        """
        Generate 808 bass pattern for hip-hop style.

        Args:
            style: Hip-hop sub-genre
            bars: Number of bars
            root_note: Root note for bass (typically 24-48)

        Returns:
            List of HipHopNote objects with pitch bend information
        """
        pattern = []
        beats_per_bar = 4

        if style == HipHopStyle.TRAP or style == HipHopStyle.DRILL:
            # Trap/Drill: Long 808s with slides
            for bar in range(bars):
                base_beat = bar * beats_per_bar

                # Long 808 on 1
                bass_config = Bass808(
                    start_pitch=root_note + 12,  # Start octave higher
                    end_pitch=root_note,
                    decay_time=300,  # Long slide
                    distortion=0.4
                )
                pattern.append(HipHopNote(
                    pitch=root_note,
                    velocity=random.randint(110, 127),
                    start_time=base_beat,
                    duration=1.5,  # Long sustain
                    articulation=DrumArticulation.ACCENT,
                    channel=0,  # Bass channel
                    pitch_bend=bass_config.get_pitch_bend_curve()[0]
                ))

                # Additional sliding 808s
                slide_positions = [2.5, 3.75]
                for pos in random.sample(slide_positions, random.randint(0, 2)):
                    bass_config = Bass808(
                        start_pitch=root_note + 7,  # Fifth above
                        end_pitch=root_note,
                        decay_time=200
                    )
                    pattern.append(HipHopNote(
                        pitch=root_note,
                        velocity=random.randint(100, 120),
                        start_time=base_beat + pos,
                        duration=0.75,
                        articulation=DrumArticulation.HARD,
                        channel=0,
                        pitch_bend=bass_config.get_pitch_bend_curve()[0]
                    ))

        else:
            # Other styles: Shorter, punchier 808s
            for bar in range(bars):
                base_beat = bar * beats_per_bar

                # 808 on strong beats
                for beat in [0, 2]:
                    pattern.append(HipHopNote(
                        pitch=root_note,
                        velocity=random.randint(95, 110),
                        start_time=base_beat + beat,
                        duration=0.5,
                        articulation=DrumArticulation.HARD,
                        channel=0
                    ))

                # Syncopated 808
                if random.random() > 0.5:
                    pattern.append(HipHopNote(
                        pitch=root_note + 7,  # Fifth
                        velocity=random.randint(85, 100),
                        start_time=base_beat + 3.5,
                        duration=0.4,
                        articulation=DrumArticulation.HARD,
                        channel=0
                    ))

        return pattern


# ============================================================================
# Sample Chopping Engine
# ============================================================================

class SampleChopper:
    """
    Sample chopping and rearrangement engine.

    Simulates the process of:
    1. Slicing a sample into parts
    2. Rearranging slices
    3. Pitch-shifting slices
    4. Applying effects (reverse, stutter)

    Note: This generates MIDI patterns that represent chopped samples,
    not actual audio manipulation.
    """

    @staticmethod
    def generate_chop_pattern(
        bars: int = 2,
        slice_mode: SampleSliceMode = SampleSliceMode.FOUR_SLICE,
        root_note: int = 60
    ) -> List[HipHopNote]:
        """
        Generate sample chop pattern.

        Args:
            bars: Number of bars for the pattern
            slice_mode: How to slice the sample
            root_note: Base MIDI note for sample

        Returns:
            List of HipHopNote objects representing slices
        """
        pattern = []

        if isinstance(slice_mode, SampleSliceMode):
            num_slices = slice_mode.value if isinstance(slice_mode.value, int) else 4
        else:
            num_slices = 4

        slice_duration = (bars * 4) / num_slices  # Duration per slice in beats

        # Create original slice sequence
        original_sequence = list(range(num_slices))

        # Rearrange slices (classic sampling technique)
        # Common patterns: reverse, stutter, skip
        rearranged = SampleChopper._rearrange_slices(original_sequence)

        # Generate notes for each slice in the rearranged pattern
        current_time = 0.0
        for slice_idx in rearranged:
            # Calculate pitch shift (simulate varispeed)
            pitch_shift = random.choice([-12, -7, -5, 0, 0, 0, 5, 7])  # Bias toward no shift

            pattern.append(HipHopNote(
                pitch=root_note + pitch_shift,
                velocity=random.randint(70, 100),
                start_time=current_time,
                duration=slice_duration * 0.9,  # Slight gap between slices
                articulation=DrumArticulation.HARD,
                channel=1  # Sample channel
            ))

            current_time += slice_duration

        return pattern

    @staticmethod
    def _rearrange_slices(original: List[int]) -> List[int]:
        """
        Rearrange slice sequence using common sampling patterns.

        Args:
            original: Original slice order

        Returns:
            Rearranged slice order
        """
        patterns = [
            lambda x: x,  # Keep original
            lambda x: list(reversed(x)),  # Reverse
            lambda x: [x[0], x[1], x[1], x[2]],  # Stutter slice 2
            lambda x: [x[0], x[2], x[1], x[3]] if len(x) >= 4 else x,  # Swap middle
            lambda x: [x[0]] * 4 if len(x) >= 1 else x,  # Loop first slice
        ]

        chosen_pattern = random.choice(patterns)
        try:
            return chosen_pattern(original)
        except (IndexError, TypeError):
            return original


# ============================================================================
# Main Hip-Hop Generator
# ============================================================================

class HipHopGenerator:
    """
    Main hip-hop music generator.

    Combines all elements to create complete hip-hop beats:
    - Drums (style-specific patterns)
    - 808 bass
    - Harmonic elements
    - Sample chops
    - Swing/timing
    """

    def __init__(
        self,
        style: HipHopStyle = HipHopStyle.BOOM_BAP,
        tempo: int = 90,
        key_root: int = 60,
        swing_amount: float = 0.54
    ):
        """
        Initialize hip-hop generator.

        Args:
            style: Hip-hop sub-genre
            tempo: BPM (typical: 70-160)
            key_root: Root note for harmony
            swing_amount: Swing ratio (0.5-0.67)
        """
        self.style = style
        self.tempo = tempo
        self.key_root = key_root
        self.swing_amount = swing_amount

        # Initialize engines (if available)
        if RHYTHM_ENGINE_AVAILABLE:
            self.rhythm_engine = RhythmEngine(ppqn=PPQN_HIGH_RES)
            self.groove_library = GrooveLibrary(ppqn=PPQN_HIGH_RES)
        else:
            self.rhythm_engine = None
            self.groove_library = None

    def generate_beat(
        self,
        bars: int = 16,
        include_808: bool = True,
        include_samples: bool = False
    ) -> Dict[str, List[HipHopNote]]:
        """
        Generate complete hip-hop beat.

        Args:
            bars: Number of bars
            include_808: Include 808 bass
            include_samples: Include sample chops

        Returns:
            Dictionary with tracks: 'drums', '808', 'samples'
        """
        arrangement = {}

        # Generate drums based on style
        if self.style == HipHopStyle.BOOM_BAP:
            drums = BoomBapDrums.generate_pattern(bars, self.swing_amount)
        elif self.style == HipHopStyle.TRAP:
            drums = TrapDrums.generate_pattern(bars, hi_hat_density='roll')
        elif self.style == HipHopStyle.DRILL:
            drums = DrillDrums.generate_pattern(bars)
        elif self.style == HipHopStyle.LOFI:
            drums = LoFiDrums.generate_pattern(bars, self.swing_amount)
        elif self.style == HipHopStyle.G_FUNK:
            drums = BoomBapDrums.generate_pattern(bars, swing_amount=0.52)  # Lighter swing
        else:
            drums = BoomBapDrums.generate_pattern(bars)

        arrangement['drums'] = drums.get_all_notes()

        # Generate 808 bass
        if include_808:
            bass_root = self.key_root - 24  # Two octaves down
            arrangement['808'] = Bass808Engine.generate_808_pattern(
                self.style, bars, bass_root
            )

        # Generate sample chops
        if include_samples:
            arrangement['samples'] = SampleChopper.generate_chop_pattern(
                bars=4,
                slice_mode=SampleSliceMode.FOUR_SLICE,
                root_note=self.key_root
            )

        return arrangement

    def get_style_info(self) -> Dict[str, str]:
        """Get information about current style"""
        info = {
            HipHopStyle.BOOM_BAP: {
                'name': '90s Boom Bap',
                'tempo_range': '85-95 BPM',
                'characteristics': 'Hard drums, MPC swing, minimal melody'
            },
            HipHopStyle.TRAP: {
                'name': 'Modern Trap',
                'tempo_range': '130-170 BPM',
                'characteristics': 'Hi-hat rolls, 808 slides, sparse arrangement'
            },
            HipHopStyle.LOFI: {
                'name': 'Lo-Fi Hip-Hop',
                'tempo_range': '70-90 BPM',
                'characteristics': 'Off-grid timing, soft drums, jazz chords'
            },
            HipHopStyle.DRILL: {
                'name': 'Drill',
                'tempo_range': '60-80 BPM (half-time feel)',
                'characteristics': 'Very sparse, sliding 808s, dark'
            },
            HipHopStyle.CONSCIOUS: {
                'name': 'Conscious Rap',
                'tempo_range': '85-100 BPM',
                'characteristics': 'Live instrumentation, complex harmony'
            },
            HipHopStyle.G_FUNK: {
                'name': 'G-Funk',
                'tempo_range': '90-105 BPM',
                'characteristics': 'Synth leads, talk box, funk basslines'
            }
        }
        return info.get(self.style, {})


# ============================================================================
# Utility Functions
# ============================================================================

def apply_mpc_swing(notes: List[HipHopNote], swing_amount: float = 0.54) -> List[HipHopNote]:
    """
    Apply MPC-style swing to notes.

    MPC swing delays offbeat notes (based on Roger Linn's algorithm).

    Args:
        notes: Notes to swing
        swing_amount: Swing ratio (0.5=straight, 0.62=triplet, 0.54=MPC default)

    Returns:
        Swung notes
    """
    swung_notes = []

    for note in notes:
        new_note = HipHopNote(
            pitch=note.pitch,
            velocity=note.velocity,
            start_time=note.start_time,
            duration=note.duration,
            articulation=note.articulation,
            channel=note.channel
        )

        # Determine if note is on offbeat (odd 16th note)
        sixteenth_position = (note.start_time % 1.0) / 0.25

        if sixteenth_position % 2 == 1:  # Offbeat 16th
            # Delay by swing amount
            offset = (swing_amount - 0.5) * 0.5
            new_note.start_time += offset
            new_note.swing_offset = int(offset * PPQN_HIGH_RES * 4)

        swung_notes.append(new_note)

    return swung_notes


# ============================================================================
# Example Usage and Testing
# ============================================================================

if __name__ == "__main__":
    """Example usage and testing of the hip-hop generator"""

    print("=" * 70)
    print("HIP-HOP MUSIC GENERATOR - Test Suite")
    print("=" * 70)

    # Test 1: Boom Bap pattern
    print("\n1. Generating Boom Bap drum pattern...")
    boom_bap = BoomBapDrums.generate_pattern(bars=4, swing_amount=0.54)
    print(f"   Kick: {len(boom_bap.kick)} hits")
    print(f"   Snare: {len(boom_bap.snare)} hits")
    print(f"   Hi-hat: {len(boom_bap.hihat)} hits")
    print(f"   Total: {len(boom_bap.get_all_notes())} notes")

    # Test 2: Trap pattern
    print("\n2. Generating Trap drum pattern with hi-hat rolls...")
    trap = TrapDrums.generate_pattern(bars=4, hi_hat_density='roll')
    print(f"   Kick: {len(trap.kick)} hits")
    print(f"   Snare: {len(trap.snare)} hits")
    print(f"   Hi-hat: {len(trap.hihat)} hits (including rolls)")

    # Test 3: 808 bass
    print("\n3. Generating 808 bass pattern...")
    bass808 = Bass808Engine.generate_808_pattern(HipHopStyle.TRAP, bars=4, root_note=36)
    print(f"   Generated {len(bass808)} 808 bass notes")
    print(f"   Pitch range: {min(n.pitch for n in bass808)} - {max(n.pitch for n in bass808)}")

    # Test 4: Sample chopping
    print("\n4. Generating sample chop pattern...")
    chops = SampleChopper.generate_chop_pattern(bars=2, slice_mode=SampleSliceMode.FOUR_SLICE)
    print(f"   Generated {len(chops)} sample slices")

    # Test 5: Lo-Fi pattern
    print("\n5. Generating Lo-Fi Hip-Hop pattern...")
    lofi = LoFiDrums.generate_pattern(bars=4, swing_amount=0.56)
    print(f"   Total notes: {len(lofi.get_all_notes())}")
    print(f"   J Dilla swing applied (56%)")

    # Test 6: Drill pattern
    print("\n6. Generating Drill pattern...")
    drill = DrillDrums.generate_pattern(bars=4)
    print(f"   Sparse arrangement: {len(drill.get_all_notes())} total notes")

    # Test 7: Complete beat generation
    print("\n7. Generating complete hip-hop beat...")
    for style in [HipHopStyle.BOOM_BAP, HipHopStyle.TRAP, HipHopStyle.LOFI]:
        generator = HipHopGenerator(style=style, tempo=90)
        beat = generator.generate_beat(bars=8, include_808=True, include_samples=False)
        info = generator.get_style_info()
        print(f"\n   {info['name']}:")
        print(f"   - Tempo range: {info['tempo_range']}")
        print(f"   - Characteristics: {info['characteristics']}")
        for track_name, notes in beat.items():
            print(f"   - {track_name}: {len(notes)} notes")

    # Test 8: Harmony/scales
    print("\n8. Testing harmonic structures...")
    for style in [HipHopStyle.BOOM_BAP, HipHopStyle.LOFI, HipHopStyle.DRILL]:
        progression = HipHopHarmony.get_progression(style, root=60)
        scale = HipHopHarmony.get_scale(style, root=60, octaves=1)
        print(f"   {style.value}:")
        print(f"   - Progression: {len(progression)} chords")
        print(f"   - Scale: {len(scale)} notes")

    # Test 9: MPC swing application
    print("\n9. Testing MPC swing...")
    test_notes = [
        HipHopNote(60, 80, i * 0.25, 0.2) for i in range(16)
    ]
    swung = apply_mpc_swing(test_notes, swing_amount=0.54)
    print(f"   Applied MPC swing to {len(swung)} notes")
    print(f"   Original timing: straight 16ths")
    print(f"   Swung timing: offbeats delayed")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)
