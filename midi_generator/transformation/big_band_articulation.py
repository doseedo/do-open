#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Big Band Articulation Engine - Agent 8 Implementation
======================================================

Implements realistic brass and woodwind articulations with MIDI pitch bend encoding
for authentic big band performance. Designed for Duke Ellington, Count Basie, and
modern big band styles.

Key Features:
-------------
- Time-varying pitch bends (falls, doits, rips, shakes, scoops)
- Style-specific articulation profiles (Ellington, Basie, Thad Jones)
- Automatic articulation suggestion based on musical context
- Full MIDI message generation with pitch bend automation
- Integration with existing ArticulationEngine

Articulation Types:
-------------------
- FALL_SHORT: Quick pitch drop at note end (-200 cents, 300ms)
- FALL_LONG: Extended pitch drop (-400 cents, 600ms) - Ellington signature
- DOIT: Quick upward scoop (+200 cents, 200ms)
- RIP: Fast ascending glissando into note (-1200→0 cents, 400ms)
- SHAKE: Rapid pitch oscillation (±100 cents @ 6Hz) - for sustained notes
- GROWL: Distortion/multiphonic effect - Ellington "jungle sound"
- SCOOP: Subtle upward approach to pitch (-100→0 cents, 150ms)
- PLUNGER: Wah-wah effect with pitch/timbre modulation

Research References:
--------------------
- Duke Ellington: "Concerto for Cootie", "Ko-Ko", "Caravan"
  - Plunger mutes (Bubber Miley, Cootie Williams)
  - Falls: typically -200 to -400 cents over 300-600ms
  - Growls: singing while playing

- Count Basie: "One O'Clock Jump", "April in Paris"
  - Shorter, crisper articulations
  - Staccato preference over long falls
  - Punchy section hits

- Thad Jones: "A Child is Born", "Three and One"
  - Modern articulation vocabulary
  - Wider intervals, less traditional jazz effects

- MIDI Implementation:
  - Pitch bend: 14-bit value (0-16383, 8192=center)
  - Pitch bend range: typically ±2 semitones (200 cents)
  - For -200 cents: 8192 - (200/200 * 4096) = 4096
  - For +200 cents: 8192 + (200/200 * 4096) = 12288

Author: Agent 8 - Articulation & Expression Engine
Date: 2025
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import math

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from midi.articulation_engine import ArticulationType as BaseArticulationType


# ============================================================================
# ENHANCED ARTICULATION TYPES FOR BIG BAND
# ============================================================================

class BigBandArticulationType(Enum):
    """Enhanced articulation types for authentic big band performance."""

    # ========== STANDARD ARTICULATIONS ==========
    NORMAL = "normal"
    STACCATO = "staccato"
    ACCENT = "accent"
    LEGATO = "legato"
    TENUTO = "tenuto"
    MARCATO = "marcato"

    # ========== JAZZ-SPECIFIC ARTICULATIONS ==========
    GHOST = "ghost"                  # Very soft, barely audible note
    SWELL = "swell"                  # Crescendo-diminuendo on single note

    # ========== PITCH BEND ARTICULATIONS (New!) ==========
    FALL_SHORT = "fall_short"        # Quick drop: -200 cents, 300ms
    FALL_LONG = "fall_long"          # Extended drop: -400 cents, 600ms
    DOIT = "doit"                    # Quick rise: +200 cents, 200ms
    RIP = "rip"                      # Fast gliss up: -1200→0 cents, 400ms
    SHAKE = "shake"                  # Rapid oscillation: ±100 cents @ 6Hz
    SCOOP = "scoop"                  # Subtle approach: -100→0 cents, 150ms
    GROWL = "growl"                  # Distortion/multiphonic effect
    PLUNGER = "plunger"              # Wah-wah with pitch modulation

    # ========== MUTE TYPES ==========
    CUP_MUTE = "cup_mute"            # Soft, mellow
    HARMON_MUTE = "harmon_mute"      # Miles Davis sound
    STRAIGHT_MUTE = "straight_mute"  # Bright, pinched


# ============================================================================
# MIDI PITCH BEND MESSAGE DATACLASS
# ============================================================================

@dataclass
class PitchBendMessage:
    """Pitch bend MIDI message with timing."""
    time_ticks: int              # Absolute time in MIDI ticks
    pitch_bend_value: int        # 14-bit value (0-16383, 8192=center)
    channel: int = 0             # MIDI channel

    def to_cents(self, pitch_bend_range: int = 200) -> float:
        """Convert pitch bend value to cents.

        Args:
            pitch_bend_range: Pitch bend range in cents (default ±200 cents = ±2 semitones)

        Returns:
            Pitch deviation in cents
        """
        # Center is 8192, full range is ±8192
        deviation_14bit = self.pitch_bend_value - 8192
        # Convert to cents based on pitch bend range
        cents = (deviation_14bit / 8192.0) * pitch_bend_range
        return cents


@dataclass
class MIDIArticulationResult:
    """Result of articulation encoding with MIDI messages."""
    notes: List[int]                      # Modified MIDI note numbers
    durations: List[float]                # Modified durations in beats
    velocities: List[int]                 # Modified velocities
    pitch_bends: List[PitchBendMessage]   # Pitch bend automation
    cc_messages: List[Dict]               # Additional CC messages (filter, mod, etc.)

    def get_all_messages_sorted(self) -> List[Union[Dict, PitchBendMessage]]:
        """Get all MIDI messages sorted by time."""
        all_msgs = []
        all_msgs.extend(self.pitch_bends)
        all_msgs.extend(self.cc_messages)
        return sorted(all_msgs, key=lambda m: m.time_ticks if hasattr(m, 'time_ticks') else m['time_ticks'])


# ============================================================================
# ARTICULATION SPECIFICATION WITH PITCH BEND
# ============================================================================

@dataclass
class BigBandArticulationSpec:
    """Specification for big band articulation with pitch bend encoding."""
    articulation: BigBandArticulationType
    note_length_multiplier: float          # 1.0 = full length
    velocity_offset: int                   # Added to velocity
    velocity_multiplier: float             # Multiplied with velocity

    # Pitch bend parameters (None if no pitch bend)
    pitch_bend_type: Optional[str] = None  # "fall", "rise", "rip", "shake", "scoop"
    pitch_bend_start_cents: Optional[int] = None  # Starting pitch deviation in cents
    pitch_bend_end_cents: Optional[int] = None    # Ending pitch deviation in cents
    pitch_bend_duration_ms: Optional[int] = None  # Duration of pitch bend
    pitch_bend_curve: str = "linear"       # "linear", "exponential", "logarithmic"

    # Shake parameters (for tremolo/shake effects)
    shake_rate_hz: Optional[float] = None  # Oscillation frequency
    shake_depth_cents: Optional[int] = None  # Oscillation amplitude

    # CC modulations
    cc_modulations: Dict[int, int] = None  # CC number -> value

    description: str = ""


# ============================================================================
# BIG BAND ARTICULATION DATABASE
# ============================================================================

BIG_BAND_ARTICULATIONS: Dict[BigBandArticulationType, BigBandArticulationSpec] = {

    # ========== STANDARD ARTICULATIONS ==========

    BigBandArticulationType.NORMAL: BigBandArticulationSpec(
        articulation=BigBandArticulationType.NORMAL,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        description="Normal articulation"
    ),

    BigBandArticulationType.STACCATO: BigBandArticulationSpec(
        articulation=BigBandArticulationType.STACCATO,
        note_length_multiplier=0.5,
        velocity_offset=5,
        velocity_multiplier=1.1,
        description="Short, detached (Basie style)"
    ),

    BigBandArticulationType.ACCENT: BigBandArticulationSpec(
        articulation=BigBandArticulationType.ACCENT,
        note_length_multiplier=1.0,
        velocity_offset=20,
        velocity_multiplier=1.2,
        description="Emphasized attack"
    ),

    BigBandArticulationType.LEGATO: BigBandArticulationSpec(
        articulation=BigBandArticulationType.LEGATO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        description="Smooth, connected"
    ),

    BigBandArticulationType.TENUTO: BigBandArticulationSpec(
        articulation=BigBandArticulationType.TENUTO,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        description="Full length, slightly emphasized"
    ),

    BigBandArticulationType.MARCATO: BigBandArticulationSpec(
        articulation=BigBandArticulationType.MARCATO,
        note_length_multiplier=0.75,
        velocity_offset=15,
        velocity_multiplier=1.3,
        description="Stressed, accented"
    ),

    # ========== JAZZ-SPECIFIC ==========

    BigBandArticulationType.GHOST: BigBandArticulationSpec(
        articulation=BigBandArticulationType.GHOST,
        note_length_multiplier=0.6,
        velocity_offset=-30,
        velocity_multiplier=0.5,
        description="Very soft, barely audible"
    ),

    BigBandArticulationType.SWELL: BigBandArticulationSpec(
        articulation=BigBandArticulationType.SWELL,
        note_length_multiplier=1.0,
        velocity_offset=10,
        velocity_multiplier=1.1,
        description="Crescendo-diminuendo on single note"
    ),

    # ========== PITCH BEND ARTICULATIONS ==========

    BigBandArticulationType.FALL_SHORT: BigBandArticulationSpec(
        articulation=BigBandArticulationType.FALL_SHORT,
        note_length_multiplier=0.8,
        velocity_offset=10,
        velocity_multiplier=1.2,
        pitch_bend_type="fall",
        pitch_bend_start_cents=0,
        pitch_bend_end_cents=-200,
        pitch_bend_duration_ms=300,
        pitch_bend_curve="exponential",  # Falls accelerate
        description="Quick pitch drop at end (-200 cents, 300ms)"
    ),

    BigBandArticulationType.FALL_LONG: BigBandArticulationSpec(
        articulation=BigBandArticulationType.FALL_LONG,
        note_length_multiplier=0.9,
        velocity_offset=10,
        velocity_multiplier=1.2,
        pitch_bend_type="fall",
        pitch_bend_start_cents=0,
        pitch_bend_end_cents=-400,
        pitch_bend_duration_ms=600,
        pitch_bend_curve="exponential",
        description="Extended pitch drop (Ellington signature: -400 cents, 600ms)"
    ),

    BigBandArticulationType.DOIT: BigBandArticulationSpec(
        articulation=BigBandArticulationType.DOIT,
        note_length_multiplier=0.7,
        velocity_offset=15,
        velocity_multiplier=1.3,
        pitch_bend_type="rise",
        pitch_bend_start_cents=0,
        pitch_bend_end_cents=200,
        pitch_bend_duration_ms=200,
        pitch_bend_curve="exponential",
        description="Quick upward pitch at end (+200 cents, 200ms)"
    ),

    BigBandArticulationType.RIP: BigBandArticulationSpec(
        articulation=BigBandArticulationType.RIP,
        note_length_multiplier=0.6,
        velocity_offset=20,
        velocity_multiplier=1.4,
        pitch_bend_type="rip",
        pitch_bend_start_cents=-1200,  # Start one octave below
        pitch_bend_end_cents=0,
        pitch_bend_duration_ms=400,
        pitch_bend_curve="logarithmic",  # Rips decelerate
        description="Fast ascending glissando into note (-1200→0 cents, 400ms)"
    ),

    BigBandArticulationType.SCOOP: BigBandArticulationSpec(
        articulation=BigBandArticulationType.SCOOP,
        note_length_multiplier=1.0,
        velocity_offset=0,
        velocity_multiplier=1.0,
        pitch_bend_type="scoop",
        pitch_bend_start_cents=-100,
        pitch_bend_end_cents=0,
        pitch_bend_duration_ms=150,
        pitch_bend_curve="linear",
        description="Subtle upward approach to pitch (-100→0 cents, 150ms)"
    ),

    BigBandArticulationType.SHAKE: BigBandArticulationSpec(
        articulation=BigBandArticulationType.SHAKE,
        note_length_multiplier=1.0,
        velocity_offset=5,
        velocity_multiplier=1.1,
        pitch_bend_type="shake",
        shake_rate_hz=6.0,  # 6 oscillations per second
        shake_depth_cents=100,
        description="Rapid pitch oscillation (±100 cents @ 6Hz)"
    ),

    BigBandArticulationType.GROWL: BigBandArticulationSpec(
        articulation=BigBandArticulationType.GROWL,
        note_length_multiplier=1.0,
        velocity_offset=10,
        velocity_multiplier=1.2,
        cc_modulations={1: 100},  # Modulation wheel for growl
        description="Distortion/multiphonic (Ellington jungle sound)"
    ),

    BigBandArticulationType.PLUNGER: BigBandArticulationSpec(
        articulation=BigBandArticulationType.PLUNGER,
        note_length_multiplier=1.0,
        velocity_offset=5,
        velocity_multiplier=1.1,
        pitch_bend_type="shake",
        shake_rate_hz=4.0,  # Slower wah-wah
        shake_depth_cents=50,
        cc_modulations={74: 80},  # Filter cutoff for wah
        description="Plunger mute wah-wah (Ellington signature)"
    ),
}


# ============================================================================
# PITCH BEND CURVE FUNCTIONS
# ============================================================================

class PitchBendCurves:
    """Generate pitch bend value sequences for various curves."""

    @staticmethod
    def linear(start_cents: int, end_cents: int, num_points: int,
               pitch_bend_range: int = 200) -> List[int]:
        """Linear pitch bend curve.

        Args:
            start_cents: Starting pitch in cents
            end_cents: Ending pitch in cents
            num_points: Number of pitch bend points
            pitch_bend_range: Pitch bend range in cents (default ±200)

        Returns:
            List of 14-bit pitch bend values
        """
        values = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            cents = start_cents + (end_cents - start_cents) * t
            pb_value = PitchBendCurves._cents_to_pitch_bend(cents, pitch_bend_range)
            values.append(pb_value)
        return values

    @staticmethod
    def exponential(start_cents: int, end_cents: int, num_points: int,
                   pitch_bend_range: int = 200) -> List[int]:
        """Exponential pitch bend curve (accelerating).

        Good for falls and doits that accelerate.
        """
        values = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Exponential curve: starts slow, accelerates
            t_curve = t * t
            cents = start_cents + (end_cents - start_cents) * t_curve
            pb_value = PitchBendCurves._cents_to_pitch_bend(cents, pitch_bend_range)
            values.append(pb_value)
        return values

    @staticmethod
    def logarithmic(start_cents: int, end_cents: int, num_points: int,
                   pitch_bend_range: int = 200) -> List[int]:
        """Logarithmic pitch bend curve (decelerating).

        Good for rips that decelerate toward target pitch.
        """
        values = []
        for i in range(num_points):
            t = i / (num_points - 1) if num_points > 1 else 0
            # Logarithmic curve: starts fast, decelerates
            t_curve = math.sqrt(t)
            cents = start_cents + (end_cents - start_cents) * t_curve
            pb_value = PitchBendCurves._cents_to_pitch_bend(cents, pitch_bend_range)
            values.append(pb_value)
        return values

    @staticmethod
    def shake(duration_ms: int, rate_hz: float, depth_cents: int,
             sample_rate_ms: int = 20, pitch_bend_range: int = 200) -> List[int]:
        """Generate shake/tremolo pitch bend curve.

        Args:
            duration_ms: Total duration in milliseconds
            rate_hz: Oscillation frequency in Hz
            depth_cents: Oscillation depth in cents (peak-to-peak)
            sample_rate_ms: Sample rate for pitch bend messages
            pitch_bend_range: Pitch bend range in cents

        Returns:
            List of 14-bit pitch bend values
        """
        num_points = max(2, duration_ms // sample_rate_ms)
        values = []

        for i in range(num_points):
            t_ms = i * sample_rate_ms
            t_sec = t_ms / 1000.0
            # Sine wave oscillation
            phase = 2 * math.pi * rate_hz * t_sec
            cents = depth_cents * math.sin(phase)
            pb_value = PitchBendCurves._cents_to_pitch_bend(cents, pitch_bend_range)
            values.append(pb_value)

        return values

    @staticmethod
    def _cents_to_pitch_bend(cents: float, pitch_bend_range: int = 200) -> int:
        """Convert cents to 14-bit pitch bend value.

        Args:
            cents: Pitch deviation in cents
            pitch_bend_range: Pitch bend range in cents (default ±200)

        Returns:
            14-bit pitch bend value (0-16383, center=8192)
        """
        # Normalize to range: cents / pitch_bend_range gives value in [-1, 1]
        normalized = cents / pitch_bend_range
        # Convert to 14-bit: center (8192) ± 8192
        pb_value = int(8192 + normalized * 8192)
        # Clamp to valid range
        return max(0, min(16383, pb_value))


# ============================================================================
# BIG BAND ARTICULATION ENGINE
# ============================================================================

class BigBandArticulationEngine:
    """
    Engine for applying big band articulations with pitch bend encoding.

    Features:
    - Time-varying pitch bends (falls, doits, rips, shakes, scoops)
    - Style-specific articulation profiles
    - Automatic articulation suggestion
    - Full MIDI message generation
    """

    def __init__(self, ticks_per_beat: int = 480, tempo_bpm: int = 120):
        """Initialize big band articulation engine.

        Args:
            ticks_per_beat: MIDI ticks per quarter note (default 480)
            tempo_bpm: Tempo in beats per minute (default 120)
        """
        self.ticks_per_beat = ticks_per_beat
        self.tempo_bpm = tempo_bpm
        self.articulation_db = BIG_BAND_ARTICULATIONS

    def apply_articulation(
        self,
        notes: List[int],
        durations: List[float],
        velocities: List[int],
        start_times: List[float],
        articulation: BigBandArticulationType,
        channel: int = 0
    ) -> MIDIArticulationResult:
        """Apply articulation to notes with full MIDI encoding.

        Args:
            notes: MIDI note numbers
            durations: Note durations in beats
            velocities: MIDI velocities
            start_times: Note start times in beats
            articulation: Articulation type to apply
            channel: MIDI channel

        Returns:
            MIDIArticulationResult with modified notes and MIDI messages
        """
        if articulation not in self.articulation_db:
            print(f"Warning: Unknown articulation {articulation}, using normal")
            articulation = BigBandArticulationType.NORMAL

        spec = self.articulation_db[articulation]

        # Modify durations
        new_durations = [dur * spec.note_length_multiplier for dur in durations]

        # Modify velocities
        new_velocities = [self._adjust_velocity(vel, spec) for vel in velocities]

        # Generate pitch bend messages
        pitch_bends = []
        if spec.pitch_bend_type:
            for i, (note, dur, start_time) in enumerate(zip(notes, new_durations, start_times)):
                bends = self._generate_pitch_bends(spec, start_time, dur, channel)
                pitch_bends.extend(bends)

        # Generate CC messages
        cc_messages = []
        if spec.cc_modulations:
            for i, start_time in enumerate(start_times):
                for cc_num, cc_val in spec.cc_modulations.items():
                    cc_messages.append({
                        'time_ticks': int(start_time * self.ticks_per_beat),
                        'cc_number': cc_num,
                        'value': cc_val,
                        'channel': channel
                    })

        return MIDIArticulationResult(
            notes=notes,
            durations=new_durations,
            velocities=new_velocities,
            pitch_bends=pitch_bends,
            cc_messages=cc_messages
        )

    def _adjust_velocity(self, original_velocity: int, spec: BigBandArticulationSpec) -> int:
        """Adjust velocity based on articulation spec."""
        adjusted = int(original_velocity * spec.velocity_multiplier) + spec.velocity_offset
        return max(1, min(127, adjusted))

    def _generate_pitch_bends(
        self,
        spec: BigBandArticulationSpec,
        start_time: float,
        duration: float,
        channel: int
    ) -> List[PitchBendMessage]:
        """Generate pitch bend messages for articulation.

        Args:
            spec: Articulation specification
            start_time: Note start time in beats
            duration: Note duration in beats
            channel: MIDI channel

        Returns:
            List of PitchBendMessage objects
        """
        messages = []

        # Convert to ticks and ms
        start_ticks = int(start_time * self.ticks_per_beat)
        duration_ticks = int(duration * self.ticks_per_beat)

        if spec.pitch_bend_type == "shake":
            # Shake: oscillation for entire note duration
            duration_ms = int((duration * 60000) / self.tempo_bpm)
            pb_values = PitchBendCurves.shake(
                duration_ms,
                spec.shake_rate_hz,
                spec.shake_depth_cents
            )

            # Distribute evenly across note duration
            for i, pb_val in enumerate(pb_values):
                t = i / len(pb_values)
                tick_offset = int(t * duration_ticks)
                messages.append(PitchBendMessage(
                    time_ticks=start_ticks + tick_offset,
                    pitch_bend_value=pb_val,
                    channel=channel
                ))

            # Reset pitch bend at end
            messages.append(PitchBendMessage(
                time_ticks=start_ticks + duration_ticks,
                pitch_bend_value=8192,
                channel=channel
            ))

        elif spec.pitch_bend_type in ["fall", "rise", "scoop", "rip"]:
            # Time-varying pitch bend
            bend_duration_ms = spec.pitch_bend_duration_ms
            bend_duration_ticks = int((bend_duration_ms / 1000.0) * (self.tempo_bpm / 60.0) * self.ticks_per_beat)

            # Determine when bend starts
            if spec.pitch_bend_type in ["fall", "rise", "doit"]:
                # Fall/doit occurs at end of note
                bend_start_ticks = start_ticks + duration_ticks - bend_duration_ticks
            else:
                # Scoop/rip occurs at beginning of note
                bend_start_ticks = start_ticks

            # Generate curve
            num_points = max(5, bend_duration_ms // 20)  # Sample every 20ms

            if spec.pitch_bend_curve == "exponential":
                pb_values = PitchBendCurves.exponential(
                    spec.pitch_bend_start_cents,
                    spec.pitch_bend_end_cents,
                    num_points
                )
            elif spec.pitch_bend_curve == "logarithmic":
                pb_values = PitchBendCurves.logarithmic(
                    spec.pitch_bend_start_cents,
                    spec.pitch_bend_end_cents,
                    num_points
                )
            else:  # linear
                pb_values = PitchBendCurves.linear(
                    spec.pitch_bend_start_cents,
                    spec.pitch_bend_end_cents,
                    num_points
                )

            # Add pitch bend messages
            for i, pb_val in enumerate(pb_values):
                t = i / (num_points - 1) if num_points > 1 else 0
                tick_offset = int(t * bend_duration_ticks)
                messages.append(PitchBendMessage(
                    time_ticks=bend_start_ticks + tick_offset,
                    pitch_bend_value=pb_val,
                    channel=channel
                ))

            # Reset pitch bend after articulation
            reset_ticks = bend_start_ticks + bend_duration_ticks + 10
            messages.append(PitchBendMessage(
                time_ticks=reset_ticks,
                pitch_bend_value=8192,
                channel=channel
            ))

        return messages

    def suggest_articulation(
        self,
        context: str,
        style: str = "basie",
        position: str = "middle"  # "start", "middle", "end", "climax"
    ) -> BigBandArticulationType:
        """Suggest appropriate articulation based on musical context.

        Args:
            context: Musical context ("phrase_ending", "section_hit", "sustained", etc.)
            style: Big band style ("ellington", "basie", "thad_jones", "modern")
            position: Position in phrase

        Returns:
            Recommended BigBandArticulationType
        """
        # Phrase ending articulations
        if context == "phrase_ending" or position == "end":
            if style == "ellington":
                return BigBandArticulationType.FALL_LONG  # Ellington loves long falls
            elif style == "basie":
                return BigBandArticulationType.STACCATO  # Basie prefers crisp endings
            else:
                return BigBandArticulationType.FALL_SHORT

        # Sustained notes
        if context == "sustained" or "whole_note" in context:
            if style == "ellington":
                return BigBandArticulationType.SHAKE  # Ellington uses shakes
            else:
                return BigBandArticulationType.TENUTO

        # Section hits
        if "hit" in context or "accent" in context:
            if style == "basie":
                return BigBandArticulationType.MARCATO  # Punchy
            else:
                return BigBandArticulationType.ACCENT

        # Shout chorus entrances
        if "shout" in context or position == "climax":
            return BigBandArticulationType.RIP  # Rip into shout chorus

        # Background figures
        if "background" in context:
            return BigBandArticulationType.STACCATO

        # Default
        return BigBandArticulationType.NORMAL


# ============================================================================
# STYLE-SPECIFIC ARTICULATION PROFILES
# ============================================================================

@dataclass
class StyleArticulationProfile:
    """Articulation profile for a specific big band style."""
    style_name: str
    description: str

    # Articulation probabilities (0.0-1.0)
    fall_probability: float
    doit_probability: float
    shake_probability: float
    growl_probability: float
    plunger_probability: float
    staccato_probability: float

    # Preferred articulations
    phrase_ending_articulation: BigBandArticulationType
    sustained_note_articulation: BigBandArticulationType
    section_hit_articulation: BigBandArticulationType

    # Dynamic range
    dynamic_range: str  # "wide", "medium", "narrow"
    use_extreme_dynamics: bool


# Duke Ellington Profile
ELLINGTON_PROFILE = StyleArticulationProfile(
    style_name="Duke Ellington",
    description="Exotic, orchestral colors, plunger mutes, growls, long falls",
    fall_probability=0.6,
    doit_probability=0.2,
    shake_probability=0.3,
    growl_probability=0.4,
    plunger_probability=0.5,  # Signature sound
    staccato_probability=0.3,
    phrase_ending_articulation=BigBandArticulationType.FALL_LONG,
    sustained_note_articulation=BigBandArticulationType.SHAKE,
    section_hit_articulation=BigBandArticulationType.ACCENT,
    dynamic_range="wide",  # ppp to fff
    use_extreme_dynamics=True
)

# Count Basie Profile
BASIE_PROFILE = StyleArticulationProfile(
    style_name="Count Basie",
    description="Simple, crisp, punchy, short articulations",
    fall_probability=0.3,
    doit_probability=0.1,
    shake_probability=0.1,
    growl_probability=0.1,
    plunger_probability=0.1,
    staccato_probability=0.7,  # Very high - signature
    phrase_ending_articulation=BigBandArticulationType.STACCATO,
    sustained_note_articulation=BigBandArticulationType.TENUTO,
    section_hit_articulation=BigBandArticulationType.MARCATO,
    dynamic_range="medium",
    use_extreme_dynamics=False
)

# Thad Jones / Modern Profile
MODERN_PROFILE = StyleArticulationProfile(
    style_name="Thad Jones / Modern",
    description="Contemporary, wider intervals, less traditional effects",
    fall_probability=0.4,
    doit_probability=0.3,
    shake_probability=0.2,
    growl_probability=0.2,
    plunger_probability=0.2,
    staccato_probability=0.5,
    phrase_ending_articulation=BigBandArticulationType.FALL_SHORT,
    sustained_note_articulation=BigBandArticulationType.TENUTO,
    section_hit_articulation=BigBandArticulationType.ACCENT,
    dynamic_range="wide",
    use_extreme_dynamics=True
)


STYLE_PROFILES = {
    "ellington": ELLINGTON_PROFILE,
    "basie": BASIE_PROFILE,
    "thad_jones": MODERN_PROFILE,
    "modern": MODERN_PROFILE,
}


# ============================================================================
# MAIN (EXAMPLES/TESTS)
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("BIG BAND ARTICULATION ENGINE - AGENT 8")
    print("Pitch Bend Encoding for Authentic Big Band Performance")
    print("=" * 80)

    engine = BigBandArticulationEngine(ticks_per_beat=480, tempo_bpm=120)

    # Example notes
    notes = [60, 64, 67, 72]
    durations = [2.0, 2.0, 2.0, 4.0]
    velocities = [80, 85, 90, 95]
    start_times = [0.0, 2.0, 4.0, 6.0]

    print("\n1. Testing pitch bend articulations:")
    articulations_to_test = [
        BigBandArticulationType.FALL_SHORT,
        BigBandArticulationType.FALL_LONG,
        BigBandArticulationType.DOIT,
        BigBandArticulationType.RIP,
        BigBandArticulationType.SHAKE,
        BigBandArticulationType.SCOOP,
    ]

    for artic in articulations_to_test:
        result = engine.apply_articulation(
            notes, durations, velocities, start_times, artic
        )
        print(f"\n{artic.value}:")
        print(f"  Duration multiplier: {result.durations[0]/durations[0]:.2f}")
        print(f"  Velocity: {result.velocities[0]} (original: {velocities[0]})")
        print(f"  Pitch bend messages: {len(result.pitch_bends)}")
        if result.pitch_bends:
            print(f"  First bend: {result.pitch_bends[0].to_cents():.1f} cents")
            print(f"  Last bend: {result.pitch_bends[-1].to_cents():.1f} cents")

    print("\n2. Style-specific articulation suggestions:")
    contexts = ["phrase_ending", "sustained", "section_hit", "shout_chorus"]
    styles = ["ellington", "basie", "modern"]

    for style in styles:
        print(f"\n{style.upper()}:")
        for context in contexts:
            suggested = engine.suggest_articulation(context, style)
            print(f"  {context}: {suggested.value}")

    print("\n3. Style profiles:")
    for style_name, profile in STYLE_PROFILES.items():
        print(f"\n{profile.style_name}:")
        print(f"  {profile.description}")
        print(f"  Fall probability: {profile.fall_probability:.0%}")
        print(f"  Growl probability: {profile.growl_probability:.0%}")
        print(f"  Staccato probability: {profile.staccato_probability:.0%}")

    print("\n" + "=" * 80)
    print("BIG BAND ARTICULATION ENGINE READY!")
    print("Integration points:")
    print("  - Use with BigBandArranger for brass/sax sections")
    print("  - Integrate with MIDI export for pitch bend messages")
    print("  - Apply style profiles (--style ellington/basie/modern)")
    print("=" * 80)
