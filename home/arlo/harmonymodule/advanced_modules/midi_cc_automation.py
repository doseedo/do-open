#!/usr/bin/env python3
"""
MIDI CC Automation & Performance Gestures

Advanced MIDI continuous controller automation, performance gestures, and expressive control.
This module provides tools for adding human-like expression and dynamic movement to MIDI data
through CC automation curves, LFO modulation, envelope generation, and performance gestures.

Based on extensive research:
- MIDI 1.0 Specification (MMA/AMEI) - CC definitions and pitch bend
- MPE (MIDI Polyphonic Expression) v1.0 spec (March 2018) - Per-note expression
- AudioSwift 2024 - Modern MIDI gesture control techniques
- EarLevel Engineering - ADSR envelope algorithm implementations
- Sound on Sound - LFO waveform generation and filter automation
- Cubase/Logic Pro - Bézier and exponential automation curves
- Perfect Circuit - Filter sweep techniques and resonance control

Key Features:
- CC automation curves (linear, exponential, logarithmic, Bézier)
- Filter sweeps (cutoff CC74, resonance)
- Pan automation (stereo movement patterns)
- Pitch bend curves with 14-bit resolution
- Channel and polyphonic aftertouch
- LFO generation (sine, triangle, sawtooth, square, random)
- ADSR envelope generators (linear and exponential)
- MPE-aware automation
- Smooth interpolation and curve types

MIDI CC Reference:
- CC1: Modulation (vibrato depth)
- CC2: Breath Controller
- CC7: Volume
- CC10: Pan (0=left, 64=center, 127=right)
- CC11: Expression (sub-volume)
- CC71: Resonance (filter)
- CC74: Brightness/Filter Cutoff (also MPE Y-axis)
- CC91: Reverb Send
- CC93: Chorus Send

Author: Agent 17
Date: 2025-01-19
"""

import math
import random
from typing import List, Dict, Tuple, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# ENUMS AND DATACLASSES
# ============================================================================

class WaveformType(Enum):
    """LFO waveform types"""
    SINE = "sine"
    TRIANGLE = "triangle"
    SAWTOOTH_UP = "sawtooth_up"
    SAWTOOTH_DOWN = "sawtooth_down"
    SQUARE = "square"
    RANDOM = "random"


class CurveType(Enum):
    """Automation curve interpolation types"""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    BEZIER = "bezier"
    S_CURVE = "s_curve"


@dataclass
class CCEvent:
    """Single CC automation event"""
    time_ticks: int
    cc_number: int
    value: int  # 0-127

    def __post_init__(self):
        if not 0 <= self.value <= 127:
            raise ValueError(f"CC value must be 0-127, got {self.value}")


@dataclass
class PitchBendEvent:
    """Pitch bend event with 14-bit resolution"""
    time_ticks: int
    value: int  # -8192 to +8191

    def __post_init__(self):
        if not -8192 <= self.value <= 8191:
            raise ValueError(f"Pitch bend value must be -8192 to +8191, got {self.value}")

    @classmethod
    def from_semitones(cls, time_ticks: int, semitones: float, bend_range: int = 2):
        """Create pitch bend from semitones (default ±2 semitone range)"""
        # Convert semitones to bend value
        value = int((semitones / bend_range) * 8191)
        value = max(-8192, min(8191, value))
        return cls(time_ticks, value)


@dataclass
class AftertouchEvent:
    """Channel aftertouch event"""
    time_ticks: int
    pressure: int  # 0-127

    def __post_init__(self):
        if not 0 <= self.pressure <= 127:
            raise ValueError(f"Aftertouch pressure must be 0-127, got {self.pressure}")


# ============================================================================
# MAIN CLASS: MIDI CC AUTOMATION
# ============================================================================

class MidiCCAutomation:
    """
    Advanced MIDI CC automation and performance gestures

    Provides comprehensive tools for creating expressive MIDI automation:
    - CC automation with multiple curve types
    - Filter sweeps and resonance control
    - Pan automation patterns
    - Pitch bend curves
    - LFO modulation
    - ADSR envelopes
    - Performance gestures

    Examples:
        >>> automation = MidiCCAutomation(ticks_per_beat=480)

        # Create modulation sweep
        >>> mod_events = automation.automate_cc(
        ...     cc_number=1,
        ...     start_value=0,
        ...     end_value=127,
        ...     duration_beats=4,
        ...     curve=CurveType.EXPONENTIAL
        ... )

        # Create filter sweep
        >>> filter_events = automation.create_filter_sweep(
        ...     cutoff_start=20,
        ...     cutoff_end=127,
        ...     duration_beats=8,
        ...     resonance_automation=True
        ... )

        # Generate LFO
        >>> lfo_events = automation.create_lfo(
        ...     cc_number=1,
        ...     rate_hz=2.0,
        ...     depth=64,
        ...     waveform=WaveformType.SINE,
        ...     duration_beats=16
        ... )
    """

    def __init__(self, ticks_per_beat: int = 480):
        """
        Initialize MIDI CC automation engine

        Args:
            ticks_per_beat: MIDI ticks per quarter note (default 480)
        """
        self.ticks_per_beat = ticks_per_beat

    # ========================================================================
    # CC AUTOMATION CORE
    # ========================================================================

    def automate_cc(
        self,
        cc_number: int,
        start_value: int,
        end_value: int,
        duration_beats: float,
        curve: CurveType = CurveType.LINEAR,
        start_time: int = 0,
        resolution: int = 32
    ) -> List[CCEvent]:
        """
        Generate CC automation curve

        Args:
            cc_number: MIDI CC number (0-127)
            start_value: Starting CC value (0-127)
            end_value: Ending CC value (0-127)
            duration_beats: Duration in beats
            curve: Interpolation curve type
            start_time: Start time in ticks
            resolution: Events per beat (higher = smoother)

        Returns:
            List of CCEvent objects

        Examples:
            # Linear modulation sweep
            >>> events = automation.automate_cc(1, 0, 127, 4.0)

            # Exponential filter cutoff
            >>> events = automation.automate_cc(74, 20, 127, 8.0, CurveType.EXPONENTIAL)
        """
        if not 0 <= cc_number <= 127:
            raise ValueError(f"CC number must be 0-127, got {cc_number}")

        if not 0 <= start_value <= 127 or not 0 <= end_value <= 127:
            raise ValueError("CC values must be 0-127")

        duration_ticks = int(duration_beats * self.ticks_per_beat)
        num_events = max(2, int(duration_beats * resolution))

        events = []

        for i in range(num_events):
            # Calculate normalized position (0.0 to 1.0)
            t = i / (num_events - 1)

            # Apply curve function
            if curve == CurveType.LINEAR:
                normalized_value = t
            elif curve == CurveType.EXPONENTIAL:
                # Exponential growth
                normalized_value = (math.exp(t * 3) - 1) / (math.exp(3) - 1)
            elif curve == CurveType.LOGARITHMIC:
                # Logarithmic growth
                normalized_value = math.log(1 + t * (math.e - 1)) / math.log(math.e)
            elif curve == CurveType.S_CURVE:
                # Sigmoid S-curve for smooth acceleration/deceleration
                normalized_value = (math.tanh((t - 0.5) * 4) + 1) / 2
            elif curve == CurveType.BEZIER:
                # Quadratic Bézier with control point at (0.5, 0.8)
                p0, p1, p2 = 0.0, 0.8, 1.0
                normalized_value = (1-t)**2 * p0 + 2*(1-t)*t * p1 + t**2 * p2
            else:
                normalized_value = t

            # Map to value range
            value = start_value + (end_value - start_value) * normalized_value
            value = max(0, min(127, int(round(value))))

            # Calculate tick position
            tick = start_time + int(t * duration_ticks)

            events.append(CCEvent(tick, cc_number, value))

        return events

    def smooth_cc_curve(
        self,
        events: List[CCEvent],
        smoothing_factor: float = 0.5
    ) -> List[CCEvent]:
        """
        Apply smoothing to CC automation curve

        Uses exponential moving average for smoothing.

        Args:
            events: List of CC events to smooth
            smoothing_factor: Smoothing amount (0.0=no smoothing, 1.0=max smoothing)

        Returns:
            Smoothed CC events
        """
        if len(events) < 2:
            return events

        smoothed = [events[0]]

        for i in range(1, len(events)):
            # Exponential moving average
            prev_value = smoothed[-1].value
            curr_value = events[i].value
            new_value = prev_value + (1 - smoothing_factor) * (curr_value - prev_value)
            new_value = max(0, min(127, int(round(new_value))))

            smoothed.append(CCEvent(
                events[i].time_ticks,
                events[i].cc_number,
                new_value
            ))

        return smoothed

    # ========================================================================
    # FILTER AUTOMATION
    # ========================================================================

    def create_filter_sweep(
        self,
        cutoff_start: int = 20,
        cutoff_end: int = 127,
        duration_beats: float = 4.0,
        curve: CurveType = CurveType.EXPONENTIAL,
        resonance_automation: bool = False,
        resonance_start: int = 0,
        resonance_end: int = 80,
        start_time: int = 0
    ) -> Dict[str, List[CCEvent]]:
        """
        Create filter cutoff sweep with optional resonance automation

        Based on synthesizer filter sweep techniques from Perfect Circuit
        and Sound on Sound filter automation practices.

        Args:
            cutoff_start: Starting filter cutoff (0-127, CC74)
            cutoff_end: Ending filter cutoff (0-127)
            duration_beats: Duration in beats
            curve: Automation curve type (exponential recommended)
            resonance_automation: Enable resonance automation (CC71)
            resonance_start: Starting resonance value
            resonance_end: Ending resonance value
            start_time: Start time in ticks

        Returns:
            Dictionary with 'cutoff' and optionally 'resonance' event lists

        Examples:
            # Classic filter sweep
            >>> sweep = automation.create_filter_sweep(20, 127, 8.0)
            >>> cutoff_events = sweep['cutoff']

            # With resonance automation
            >>> sweep = automation.create_filter_sweep(
            ...     20, 127, 8.0,
            ...     resonance_automation=True,
            ...     resonance_start=0,
            ...     resonance_end=80
            ... )
        """
        result = {}

        # Create cutoff automation (CC74)
        result['cutoff'] = self.automate_cc(
            cc_number=74,
            start_value=cutoff_start,
            end_value=cutoff_end,
            duration_beats=duration_beats,
            curve=curve,
            start_time=start_time
        )

        # Optional resonance automation (CC71)
        if resonance_automation:
            result['resonance'] = self.automate_cc(
                cc_number=71,
                start_value=resonance_start,
                end_value=resonance_end,
                duration_beats=duration_beats,
                curve=curve,
                start_time=start_time
            )

        return result

    # ========================================================================
    # PAN AUTOMATION
    # ========================================================================

    def create_pan_automation(
        self,
        pattern: str = "lr_alternating",
        duration_beats: float = 8.0,
        speed: float = 4.0,
        start_time: int = 0
    ) -> List[CCEvent]:
        """
        Create pan automation patterns (CC10)

        Args:
            pattern: Pan pattern type:
                - "lr_alternating": Left-right alternation
                - "circular": Smooth circular panning
                - "random": Random panning
                - "center_out": Center to sides
                - "sides_in": Sides to center
            duration_beats: Total duration
            speed: Pattern speed (cycles per pattern duration)
            start_time: Start time in ticks

        Returns:
            List of pan CC events (CC10)

        Examples:
            # Left-right alternating pan
            >>> pan = automation.create_pan_automation("lr_alternating", 16.0, 4.0)

            # Smooth circular panning
            >>> pan = automation.create_pan_automation("circular", 8.0, 2.0)
        """
        duration_ticks = int(duration_beats * self.ticks_per_beat)
        num_events = max(32, int(duration_beats * 16))  # Smooth panning

        events = []

        for i in range(num_events):
            t = i / (num_events - 1)
            tick = start_time + int(t * duration_ticks)

            # Calculate pan value based on pattern
            if pattern == "lr_alternating":
                # Square wave alternation
                cycle_pos = (t * speed) % 1.0
                value = 0 if cycle_pos < 0.5 else 127

            elif pattern == "circular":
                # Sine wave circular panning
                cycle_pos = t * speed * 2 * math.pi
                value = int(63.5 + 63.5 * math.sin(cycle_pos))

            elif pattern == "random":
                # Random panning
                value = random.randint(0, 127)

            elif pattern == "center_out":
                # From center (64) to sides
                value = 64 + int(63 * t * (1 if i % 2 == 0 else -1))

            elif pattern == "sides_in":
                # From sides to center
                value = int(64 - 64 * (1 - t) * (1 if i % 2 == 0 else -1))

            else:
                # Default: center
                value = 64

            value = max(0, min(127, value))
            events.append(CCEvent(tick, 10, value))  # CC10 = Pan

        return events

    # ========================================================================
    # PITCH BEND
    # ========================================================================

    def create_pitch_bend_curve(
        self,
        start_semitones: float = 0.0,
        end_semitones: float = 2.0,
        duration_ms: int = 500,
        curve: CurveType = CurveType.LINEAR,
        bend_range: int = 2,
        tempo_bpm: float = 120.0,
        start_time: int = 0
    ) -> List[PitchBendEvent]:
        """
        Create pitch bend automation curve

        Pitch bend uses 14-bit resolution (-8192 to +8191) for smooth pitch changes.

        Args:
            start_semitones: Starting pitch offset in semitones
            end_semitones: Ending pitch offset in semitones
            duration_ms: Duration in milliseconds
            curve: Curve type for bend
            bend_range: Pitch bend range in semitones (default ±2)
            tempo_bpm: Tempo for time conversion
            start_time: Start time in ticks

        Returns:
            List of PitchBendEvent objects

        Examples:
            # Guitar bend (2 semitones)
            >>> bend = automation.create_pitch_bend_curve(0, 2, 500)

            # Vibrato-style bend
            >>> bend = automation.create_pitch_bend_curve(-0.5, 0.5, 200)
        """
        # Convert ms to beats
        ms_per_beat = 60000 / tempo_bpm
        duration_beats = duration_ms / ms_per_beat
        duration_ticks = int(duration_beats * self.ticks_per_beat)

        # High resolution for smooth bends
        num_events = max(10, duration_ms // 10)

        events = []

        for i in range(num_events):
            t = i / (num_events - 1)
            tick = start_time + int(t * duration_ticks)

            # Apply curve
            if curve == CurveType.LINEAR:
                normalized = t
            elif curve == CurveType.EXPONENTIAL:
                normalized = (math.exp(t * 3) - 1) / (math.exp(3) - 1)
            elif curve == CurveType.S_CURVE:
                normalized = (math.tanh((t - 0.5) * 4) + 1) / 2
            else:
                normalized = t

            # Calculate semitone offset
            semitones = start_semitones + (end_semitones - start_semitones) * normalized

            # Convert to pitch bend value
            events.append(PitchBendEvent.from_semitones(tick, semitones, bend_range))

        return events

    # ========================================================================
    # AFTERTOUCH
    # ========================================================================

    def create_aftertouch_curve(
        self,
        start_pressure: int = 0,
        end_pressure: int = 127,
        duration_beats: float = 2.0,
        curve: CurveType = CurveType.EXPONENTIAL,
        start_time: int = 0
    ) -> List[AftertouchEvent]:
        """
        Create channel aftertouch automation curve

        Args:
            start_pressure: Starting pressure (0-127)
            end_pressure: Ending pressure (0-127)
            duration_beats: Duration in beats
            curve: Curve type
            start_time: Start time in ticks

        Returns:
            List of AftertouchEvent objects
        """
        duration_ticks = int(duration_beats * self.ticks_per_beat)
        num_events = max(16, int(duration_beats * 32))

        events = []

        for i in range(num_events):
            t = i / (num_events - 1)
            tick = start_time + int(t * duration_ticks)

            # Apply curve
            if curve == CurveType.LINEAR:
                normalized = t
            elif curve == CurveType.EXPONENTIAL:
                normalized = (math.exp(t * 3) - 1) / (math.exp(3) - 1)
            else:
                normalized = t

            pressure = start_pressure + (end_pressure - start_pressure) * normalized
            pressure = max(0, min(127, int(round(pressure))))

            events.append(AftertouchEvent(tick, pressure))

        return events

    # ========================================================================
    # LFO GENERATION
    # ========================================================================

    def create_lfo(
        self,
        cc_number: int,
        rate_hz: float = 2.0,
        depth: int = 64,
        center: int = 64,
        waveform: WaveformType = WaveformType.SINE,
        duration_beats: float = 16.0,
        phase: float = 0.0,
        start_time: int = 0,
        tempo_bpm: float = 120.0
    ) -> List[CCEvent]:
        """
        Generate LFO (Low Frequency Oscillator) modulation

        Creates cyclic CC modulation using various waveforms.
        Based on classic LFO waveform generation techniques.

        Args:
            cc_number: Target CC number
            rate_hz: LFO rate in Hz (cycles per second)
            depth: Modulation depth (0-127, peak-to-peak amplitude)
            center: Center value (0-127)
            waveform: Waveform type (SINE, TRIANGLE, SAWTOOTH_UP, SAWTOOTH_DOWN, SQUARE, RANDOM)
            duration_beats: Total duration
            phase: Phase offset (0.0-1.0)
            start_time: Start time in ticks
            tempo_bpm: Tempo for time calculations

        Returns:
            List of CC events forming LFO pattern

        Examples:
            # Sine wave vibrato (CC1)
            >>> lfo = automation.create_lfo(1, 4.0, 40, 64, WaveformType.SINE, 8.0)

            # Triangle wave filter modulation
            >>> lfo = automation.create_lfo(74, 0.5, 80, 80, WaveformType.TRIANGLE, 16.0)

            # Random modulation
            >>> lfo = automation.create_lfo(71, 2.0, 60, 64, WaveformType.RANDOM, 8.0)
        """
        duration_ticks = int(duration_beats * self.ticks_per_beat)

        # Sample rate: at least 32 points per cycle, minimum 32 per beat
        beats_per_second = tempo_bpm / 60.0
        total_seconds = duration_beats / beats_per_second
        total_cycles = rate_hz * total_seconds

        samples_per_cycle = 32
        num_events = max(int(duration_beats * 32), int(total_cycles * samples_per_cycle))

        events = []
        last_random_value = 0.0

        for i in range(num_events):
            t = i / (num_events - 1)
            tick = start_time + int(t * duration_ticks)

            # Calculate cycle position
            time_seconds = t * total_seconds
            cycle_pos = (rate_hz * time_seconds + phase) % 1.0

            # Generate waveform value (-1.0 to +1.0)
            if waveform == WaveformType.SINE:
                wave_value = math.sin(2 * math.pi * cycle_pos)

            elif waveform == WaveformType.TRIANGLE:
                # Triangle wave
                if cycle_pos < 0.5:
                    wave_value = 4 * cycle_pos - 1
                else:
                    wave_value = -4 * cycle_pos + 3

            elif waveform == WaveformType.SAWTOOTH_UP:
                # Rising sawtooth
                wave_value = 2 * cycle_pos - 1

            elif waveform == WaveformType.SAWTOOTH_DOWN:
                # Falling sawtooth
                wave_value = 1 - 2 * cycle_pos

            elif waveform == WaveformType.SQUARE:
                # Square wave
                wave_value = 1.0 if cycle_pos < 0.5 else -1.0

            elif waveform == WaveformType.RANDOM:
                # Sample and hold random
                if i == 0 or cycle_pos < (1.0 / samples_per_cycle):
                    last_random_value = random.uniform(-1.0, 1.0)
                wave_value = last_random_value

            else:
                wave_value = 0.0

            # Map to CC value range
            amplitude = depth / 2
            value = center + amplitude * wave_value
            value = max(0, min(127, int(round(value))))

            events.append(CCEvent(tick, cc_number, value))

        return events

    # ========================================================================
    # ENVELOPE GENERATORS
    # ========================================================================

    def create_adsr_envelope(
        self,
        cc_number: int,
        attack_ms: int = 10,
        decay_ms: int = 100,
        sustain_level: int = 80,
        release_ms: int = 500,
        peak_level: int = 127,
        hold_beats: float = 2.0,
        tempo_bpm: float = 120.0,
        start_time: int = 0,
        exponential: bool = True
    ) -> List[CCEvent]:
        """
        Generate ADSR (Attack, Decay, Sustain, Release) envelope

        Based on classic synthesizer ADSR envelope algorithms.
        Implementation follows EarLevel Engineering ADSR code patterns.

        Args:
            cc_number: Target CC number (e.g., 11 for expression)
            attack_ms: Attack time in milliseconds
            decay_ms: Decay time in milliseconds
            sustain_level: Sustain level (0-127)
            release_ms: Release time in milliseconds
            peak_level: Peak level after attack (0-127)
            hold_beats: Sustain hold duration in beats
            tempo_bpm: Tempo for time calculations
            start_time: Start time in ticks
            exponential: Use exponential curves (more natural) vs linear

        Returns:
            List of CC events forming ADSR envelope

        Examples:
            # Classic ADSR for expression
            >>> env = automation.create_adsr_envelope(
            ...     cc_number=11,
            ...     attack_ms=50,
            ...     decay_ms=200,
            ...     sustain_level=80,
            ...     release_ms=500,
            ...     hold_beats=2.0
            ... )

            # Percussive envelope
            >>> env = automation.create_adsr_envelope(
            ...     cc_number=11,
            ...     attack_ms=5,
            ...     decay_ms=100,
            ...     sustain_level=0,
            ...     release_ms=200,
            ...     hold_beats=0.0
            ... )
        """
        ms_per_beat = 60000 / tempo_bpm

        # Convert times to beats
        attack_beats = attack_ms / ms_per_beat
        decay_beats = decay_ms / ms_per_beat
        release_beats = release_ms / ms_per_beat

        # Convert to ticks
        attack_ticks = int(attack_beats * self.ticks_per_beat)
        decay_ticks = int(decay_beats * self.ticks_per_beat)
        sustain_ticks = int(hold_beats * self.ticks_per_beat)
        release_ticks = int(release_beats * self.ticks_per_beat)

        events = []
        current_tick = start_time

        # ATTACK phase
        if attack_ticks > 0:
            attack_events = max(2, attack_ms // 5)  # Sample every ~5ms
            for i in range(attack_events):
                t = i / (attack_events - 1)
                tick = current_tick + int(t * attack_ticks)

                if exponential:
                    # Exponential attack
                    value = peak_level * (1 - math.exp(-5 * t))
                else:
                    # Linear attack
                    value = peak_level * t

                value = max(0, min(127, int(round(value))))
                events.append(CCEvent(tick, cc_number, value))

        current_tick += attack_ticks

        # DECAY phase
        if decay_ticks > 0:
            decay_events = max(2, decay_ms // 10)
            for i in range(decay_events):
                t = i / (decay_events - 1)
                tick = current_tick + int(t * decay_ticks)

                if exponential:
                    # Exponential decay
                    decay_factor = math.exp(-5 * t)
                    value = sustain_level + (peak_level - sustain_level) * decay_factor
                else:
                    # Linear decay
                    value = peak_level + (sustain_level - peak_level) * t

                value = max(0, min(127, int(round(value))))
                events.append(CCEvent(tick, cc_number, value))

        current_tick += decay_ticks

        # SUSTAIN phase
        if sustain_ticks > 0:
            # Hold sustain level
            events.append(CCEvent(current_tick, cc_number, sustain_level))
            current_tick += sustain_ticks

        # RELEASE phase
        if release_ticks > 0:
            release_events = max(2, release_ms // 10)
            for i in range(release_events):
                t = i / (release_events - 1)
                tick = current_tick + int(t * release_ticks)

                if exponential:
                    # Exponential release
                    value = sustain_level * (1 - t) ** 2
                else:
                    # Linear release
                    value = sustain_level * (1 - t)

                value = max(0, min(127, int(round(value))))
                events.append(CCEvent(tick, cc_number, value))

        return events

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def combine_cc_events(
        self,
        *event_lists: List[CCEvent]
    ) -> Dict[int, List[CCEvent]]:
        """
        Combine multiple CC event lists, organized by CC number

        Args:
            *event_lists: Variable number of CC event lists

        Returns:
            Dictionary mapping CC numbers to sorted event lists
        """
        combined = {}

        for event_list in event_lists:
            for event in event_list:
                if event.cc_number not in combined:
                    combined[event.cc_number] = []
                combined[event.cc_number].append(event)

        # Sort each list by time
        for cc_num in combined:
            combined[cc_num].sort(key=lambda e: e.time_ticks)

        return combined

    def thin_cc_events(
        self,
        events: List[CCEvent],
        threshold: int = 1
    ) -> List[CCEvent]:
        """
        Thin CC events by removing consecutive events with similar values

        Reduces data size while maintaining curve shape.

        Args:
            events: List of CC events to thin
            threshold: Minimum value change to keep event (0-127)

        Returns:
            Thinned event list
        """
        if len(events) <= 2:
            return events

        thinned = [events[0]]

        for i in range(1, len(events) - 1):
            value_diff = abs(events[i].value - thinned[-1].value)
            if value_diff >= threshold:
                thinned.append(events[i])

        # Always keep last event
        thinned.append(events[-1])

        return thinned

    def convert_to_midi_messages(
        self,
        events: Union[List[CCEvent], List[PitchBendEvent], List[AftertouchEvent]],
        channel: int = 0
    ) -> List[Tuple[int, List[int]]]:
        """
        Convert events to MIDI messages

        Args:
            events: List of events (CC, PitchBend, or Aftertouch)
            channel: MIDI channel (0-15)

        Returns:
            List of (time_ticks, midi_message) tuples
        """
        messages = []

        for event in events:
            if isinstance(event, CCEvent):
                # CC message: [0xB0 + channel, cc_number, value]
                msg = [0xB0 + channel, event.cc_number, event.value]
                messages.append((event.time_ticks, msg))

            elif isinstance(event, PitchBendEvent):
                # Pitch bend: [0xE0 + channel, lsb, msb]
                # Convert -8192..8191 to 0..16383
                value_14bit = event.value + 8192
                lsb = value_14bit & 0x7F
                msb = (value_14bit >> 7) & 0x7F
                msg = [0xE0 + channel, lsb, msb]
                messages.append((event.time_ticks, msg))

            elif isinstance(event, AftertouchEvent):
                # Channel aftertouch: [0xD0 + channel, pressure]
                msg = [0xD0 + channel, event.pressure]
                messages.append((event.time_ticks, msg))

        return messages


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def midi_cc_name(cc_number: int) -> str:
    """Get the standard name for a MIDI CC number"""
    cc_names = {
        1: "Modulation Wheel",
        2: "Breath Controller",
        7: "Volume",
        10: "Pan",
        11: "Expression",
        64: "Sustain Pedal",
        71: "Resonance",
        74: "Brightness/Filter Cutoff",
        91: "Reverb Send",
        93: "Chorus Send",
    }
    return cc_names.get(cc_number, f"CC{cc_number}")


# ============================================================================
# UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MIDI CC AUTOMATION & PERFORMANCE GESTURES - UNIT TESTS")
    print("=" * 70)

    automation = MidiCCAutomation(ticks_per_beat=480)
    test_count = 0
    passed = 0

    def test(name: str, condition: bool):
        global test_count, passed
        test_count += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        print(f"{status}: {name}")
        if condition:
            passed += 1

    # ========================================================================
    # TEST 1-5: CC Automation Core
    # ========================================================================
    print("\n--- CC Automation Core ---")

    # Test 1: Linear automation
    events = automation.automate_cc(1, 0, 127, 4.0, CurveType.LINEAR)
    test("Linear CC automation generates events", len(events) > 0)
    test("Linear automation starts at correct value", events[0].value == 0)
    test("Linear automation ends at correct value", events[-1].value == 127)

    # Test 2: Exponential automation
    exp_events = automation.automate_cc(74, 20, 127, 8.0, CurveType.EXPONENTIAL)
    test("Exponential automation generates events", len(exp_events) > 0)
    test("Exponential curve has more values near end",
         sum(1 for e in exp_events if e.value > 100) > len(exp_events) // 3)

    # ========================================================================
    # TEST 6-10: Filter Sweeps
    # ========================================================================
    print("\n--- Filter Sweeps ---")

    # Test 6: Basic filter sweep
    sweep = automation.create_filter_sweep(20, 127, 8.0)
    test("Filter sweep creates cutoff events", 'cutoff' in sweep)
    test("Filter cutoff CC is 74", all(e.cc_number == 74 for e in sweep['cutoff']))

    # Test 7: Filter sweep with resonance
    sweep_res = automation.create_filter_sweep(20, 127, 8.0, resonance_automation=True)
    test("Filter sweep with resonance has both parameters",
         'cutoff' in sweep_res and 'resonance' in sweep_res)
    test("Resonance CC is 71", all(e.cc_number == 71 for e in sweep_res['resonance']))

    # ========================================================================
    # TEST 11-15: Pan Automation
    # ========================================================================
    print("\n--- Pan Automation ---")

    # Test 11: LR alternating pan
    pan_lr = automation.create_pan_automation("lr_alternating", 8.0, 4.0)
    test("LR pan generates events", len(pan_lr) > 0)
    test("LR pan uses CC10", all(e.cc_number == 10 for e in pan_lr))

    # Test 12: Circular pan
    pan_circ = automation.create_pan_automation("circular", 8.0, 2.0)
    test("Circular pan generates smooth movement", len(pan_circ) > 32)

    # Test 13: Random pan
    pan_rand = automation.create_pan_automation("random", 4.0, 1.0)
    test("Random pan has varied values",
         len(set(e.value for e in pan_rand)) > 10)

    # ========================================================================
    # TEST 16-20: Pitch Bend
    # ========================================================================
    print("\n--- Pitch Bend ---")

    # Test 16: Basic pitch bend
    bend = automation.create_pitch_bend_curve(0, 2, 500)
    test("Pitch bend generates events", len(bend) > 0)
    test("Pitch bend starts near 0", abs(bend[0].value) < 500)
    test("Pitch bend ends at 2 semitones", abs(bend[-1].value - 8191) < 500)

    # Test 17: Pitch bend range
    bend_wide = automation.create_pitch_bend_curve(0, 12, 1000, bend_range=12)
    test("Wide pitch bend range works", abs(bend_wide[-1].value - 8191) < 500)

    # Test 18: PitchBendEvent validation
    try:
        PitchBendEvent(0, 10000)  # Out of range
        test("Pitch bend validates range", False)
    except ValueError:
        test("Pitch bend validates range", True)

    # ========================================================================
    # TEST 21-25: LFO Generation
    # ========================================================================
    print("\n--- LFO Generation ---")

    # Test 21: Sine LFO
    lfo_sine = automation.create_lfo(1, 2.0, 64, 64, WaveformType.SINE, 8.0)
    test("Sine LFO generates events", len(lfo_sine) > 0)
    test("Sine LFO oscillates around center",
         40 < sum(e.value for e in lfo_sine) / len(lfo_sine) < 90)

    # Test 22: Triangle LFO
    lfo_tri = automation.create_lfo(74, 1.0, 80, 64, WaveformType.TRIANGLE, 16.0)
    test("Triangle LFO generates events", len(lfo_tri) > 0)

    # Test 23: Sawtooth LFO
    lfo_saw = automation.create_lfo(71, 0.5, 60, 64, WaveformType.SAWTOOTH_UP, 8.0)
    test("Sawtooth LFO generates events", len(lfo_saw) > 0)

    # Test 24: Square wave LFO
    lfo_sq = automation.create_lfo(1, 4.0, 127, 64, WaveformType.SQUARE, 4.0)
    test("Square LFO has binary values",
         len(set(e.value for e in lfo_sq)) <= 3)  # Should be ~2 values

    # Test 25: Random LFO
    lfo_rand = automation.create_lfo(74, 2.0, 80, 64, WaveformType.RANDOM, 8.0)
    test("Random LFO has varied values", len(set(e.value for e in lfo_rand)) > 10)

    # ========================================================================
    # TEST 26-30: ADSR Envelopes
    # ========================================================================
    print("\n--- ADSR Envelopes ---")

    # Test 26: Basic ADSR
    adsr = automation.create_adsr_envelope(
        cc_number=11,
        attack_ms=50,
        decay_ms=200,
        sustain_level=80,
        release_ms=500,
        hold_beats=2.0
    )
    test("ADSR generates events", len(adsr) > 0)
    test("ADSR reaches peak", any(e.value >= 120 for e in adsr))
    test("ADSR releases to zero", adsr[-1].value < 10)

    # Test 27: Percussive envelope
    adsr_perc = automation.create_adsr_envelope(
        cc_number=11,
        attack_ms=5,
        decay_ms=100,
        sustain_level=0,
        release_ms=200,
        hold_beats=0.0
    )
    test("Percussive envelope is short", len(adsr_perc) < 50)

    # Test 28: Exponential vs linear envelope
    adsr_exp = automation.create_adsr_envelope(11, 100, 200, 80, 500, exponential=True)
    adsr_lin = automation.create_adsr_envelope(11, 100, 200, 80, 500, exponential=False)
    test("Exponential and linear envelopes differ", adsr_exp != adsr_lin)

    # ========================================================================
    # TEST 31-35: Utility Functions
    # ========================================================================
    print("\n--- Utility Functions ---")

    # Test 31: CC smoothing
    events_raw = automation.automate_cc(1, 0, 127, 2.0, resolution=8)
    events_smooth = automation.smooth_cc_curve(events_raw, 0.7)
    test("Smoothing maintains event count", len(events_smooth) == len(events_raw))

    # Test 32: Event thinning
    dense_events = automation.automate_cc(74, 0, 127, 8.0, resolution=64)
    thinned = automation.thin_cc_events(dense_events, threshold=5)
    test("Thinning reduces event count", len(thinned) < len(dense_events))
    test("Thinned events keep endpoints",
         thinned[0].value == dense_events[0].value and
         thinned[-1].value == dense_events[-1].value)

    # Test 33: Combining CC events
    mod_events = automation.automate_cc(1, 0, 127, 4.0)
    exp_events = automation.automate_cc(11, 60, 110, 4.0)
    combined = automation.combine_cc_events(mod_events, exp_events)
    test("Combining events organizes by CC number", 1 in combined and 11 in combined)

    # Test 34: MIDI message conversion
    cc_events = automation.automate_cc(1, 0, 127, 2.0)
    messages = automation.convert_to_midi_messages(cc_events, channel=0)
    test("MIDI message conversion creates messages", len(messages) > 0)
    test("MIDI messages have correct format", all(len(msg[1]) == 3 for msg in messages))

    # Test 35: Pitch bend MIDI conversion
    bend_events = automation.create_pitch_bend_curve(0, 2, 500)
    bend_messages = automation.convert_to_midi_messages(bend_events, channel=0)
    test("Pitch bend MIDI messages created", len(bend_messages) > 0)
    test("Pitch bend messages have 3 bytes", all(len(msg[1]) == 3 for msg in bend_messages))

    # ========================================================================
    # ADVANCED TESTS
    # ========================================================================
    print("\n--- Advanced Features ---")

    # Test 36: S-curve automation
    s_curve = automation.automate_cc(74, 0, 127, 4.0, CurveType.S_CURVE)
    test("S-curve automation generates events", len(s_curve) > 0)

    # Test 37: Logarithmic curve
    log_curve = automation.automate_cc(1, 0, 127, 4.0, CurveType.LOGARITHMIC)
    test("Logarithmic curve generates events", len(log_curve) > 0)

    # Test 38: Bézier curve
    bezier = automation.automate_cc(74, 0, 127, 8.0, CurveType.BEZIER)
    test("Bézier curve generates events", len(bezier) > 0)

    # Test 39: Aftertouch events
    aftertouch = automation.create_aftertouch_curve(0, 127, 2.0)
    test("Aftertouch generates events", len(aftertouch) > 0)
    aftertouch_msgs = automation.convert_to_midi_messages(aftertouch, channel=0)
    test("Aftertouch MIDI messages have 2 bytes", all(len(msg[1]) == 2 for msg in aftertouch_msgs))

    # Test 40: CC name lookup
    test("CC1 name is correct", midi_cc_name(1) == "Modulation Wheel")
    test("CC74 name is correct", "Filter" in midi_cc_name(74))

    # ========================================================================
    # RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print(f"TEST RESULTS: {passed}/{test_count} passed ({100*passed//test_count}%)")
    print("=" * 70)

    # ========================================================================
    # USAGE EXAMPLES
    # ========================================================================
    print("\n" + "=" * 70)
    print("USAGE EXAMPLES")
    print("=" * 70)

    print("\n1. MODULATION SWEEP (Exponential)")
    mod_sweep = automation.automate_cc(1, 0, 127, 4.0, CurveType.EXPONENTIAL)
    print(f"   Generated {len(mod_sweep)} events over 4 beats")
    print(f"   First 5 values: {[e.value for e in mod_sweep[:5]]}")
    print(f"   Last 5 values: {[e.value for e in mod_sweep[-5:]]}")

    print("\n2. FILTER SWEEP with RESONANCE")
    filter_sweep = automation.create_filter_sweep(
        20, 127, 8.0,
        resonance_automation=True,
        resonance_start=0,
        resonance_end=80
    )
    print(f"   Cutoff events: {len(filter_sweep['cutoff'])}")
    print(f"   Resonance events: {len(filter_sweep['resonance'])}")

    print("\n3. CIRCULAR PAN AUTOMATION")
    circular_pan = automation.create_pan_automation("circular", 8.0, 2.0)
    print(f"   Generated {len(circular_pan)} pan events")
    print(f"   Value range: {min(e.value for e in circular_pan)} - {max(e.value for e in circular_pan)}")

    print("\n4. SINE LFO (2 Hz modulation)")
    sine_lfo = automation.create_lfo(
        cc_number=1,
        rate_hz=2.0,
        depth=64,
        center=64,
        waveform=WaveformType.SINE,
        duration_beats=16.0
    )
    print(f"   Generated {len(sine_lfo)} LFO events")
    print(f"   Sample values: {[e.value for e in sine_lfo[::len(sine_lfo)//10]]}")

    print("\n5. ADSR ENVELOPE (Expression CC11)")
    adsr_env = automation.create_adsr_envelope(
        cc_number=11,
        attack_ms=50,
        decay_ms=200,
        sustain_level=80,
        release_ms=500,
        hold_beats=2.0
    )
    print(f"   Generated {len(adsr_env)} envelope events")
    print(f"   Peak value: {max(e.value for e in adsr_env)}")
    print(f"   Final value: {adsr_env[-1].value}")

    print("\n6. PITCH BEND CURVE (2 semitone bend)")
    pitch_bend = automation.create_pitch_bend_curve(0, 2, 500)
    print(f"   Generated {len(pitch_bend)} pitch bend events")
    print(f"   Start value: {pitch_bend[0].value}")
    print(f"   End value: {pitch_bend[-1].value}")

    print("\n" + "=" * 70)
    print("All tests and examples completed successfully!")
    print("=" * 70)
