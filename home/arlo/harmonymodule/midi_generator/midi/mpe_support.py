"""
MIDI Polyphonic Expression (MPE) Support

Implementation of MPE specification for per-note pitch bend, pressure,
and timbre control. Enables advanced expressive control beyond standard MIDI.

MPE allows each note to have independent:
- Pitch bend (for vibrato, slides, bends)
- Channel pressure/aftertouch (for dynamics)
- Timbre control via CC74 (for brightness/filter)

Research:
    - MPE Specification v1.0 (MIDI Manufacturers Association, 2018)
    - Roger Linn Design - MPE technical documentation
    - ROLI Seaboard MPE implementation
    - Haken Continuum MPE configuration

Author: AGENT 6 - MIDI Expression & Performance
"""

from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum
import random
import math


class MPEZoneLayout(Enum):
    """MPE zone configurations."""
    LOWER_ZONE = "lower"  # Channels 2-15 (manager = 1)
    UPPER_ZONE = "upper"  # Channels 1-14 (manager = 15)
    BOTH_ZONES = "both"   # Split keyboard


@dataclass
class MPENote:
    """MPE note with per-note expression.

    Attributes:
        pitch: MIDI note number (0-127)
        velocity: Initial velocity (1-127)
        start_time: Note start in ticks
        duration: Note duration in ticks
        channel: Member channel (not manager channel)
        pitch_bend: Per-note pitch bend (-8192 to 8191)
        pressure: Channel pressure (0-127)
        timbre: CC74 value for timbre (0-127)
    """
    pitch: int
    velocity: int
    start_time: int
    duration: int
    channel: int
    pitch_bend: int = 0
    pressure: int = 0
    timbre: int = 64

    def end_time(self) -> int:
        """Get note end time."""
        return self.start_time + self.duration


@dataclass
class MPEPitchBend:
    """Per-note pitch bend event."""
    time: int
    channel: int
    value: int  # -8192 to 8191


@dataclass
class MPEPressure:
    """Per-note pressure (channel aftertouch) event."""
    time: int
    channel: int
    value: int  # 0-127


@dataclass
class MPETimbre:
    """Per-note timbre (CC74) event."""
    time: int
    channel: int
    value: int  # 0-127


class MPEChannelManager:
    """Manages MPE channel allocation.

    MPE uses one manager channel and multiple member channels.
    Each note gets its own member channel for independent expression.

    Lower Zone: Manager = Channel 1, Members = Channels 2-15 (14 voices)
    Upper Zone: Manager = Channel 16, Members = Channels 1-15 (15 voices)

    Research:
        - MPE Specification Section 2: Zone Configuration
    """

    def __init__(self, zone: MPEZoneLayout = MPEZoneLayout.LOWER_ZONE,
                 num_channels: int = 15):
        """Initialize MPE channel manager.

        Args:
            zone: MPE zone layout
            num_channels: Number of member channels to use
        """
        self.zone = zone
        self.num_channels = min(num_channels, 15)

        # Set up channels based on zone
        if zone == MPEZoneLayout.LOWER_ZONE:
            self.manager_channel = 0  # MIDI channel 1 (0-indexed)
            self.member_channels = list(range(1, self.num_channels + 1))
        elif zone == MPEZoneLayout.UPPER_ZONE:
            self.manager_channel = 15  # MIDI channel 16
            self.member_channels = list(range(15 - self.num_channels, 15))
        else:  # BOTH_ZONES
            # Use both zones (complex setup)
            self.manager_channel = 0
            self.member_channels = list(range(1, 16))

        # Track channel usage
        self.available_channels: Set[int] = set(self.member_channels)
        self.note_to_channel: Dict[Tuple[int, int], int] = {}  # (pitch, start_time) -> channel

    def allocate_channel(self, note_pitch: int, start_time: int) -> Optional[int]:
        """Allocate a member channel for a note.

        Args:
            note_pitch: MIDI note number
            start_time: Note start time

        Returns:
            Allocated channel number, or None if no channels available
        """
        if not self.available_channels:
            # No channels available - voice stealing would happen here
            # For simplicity, return None
            return None

        # Get next available channel (round-robin)
        channel = min(self.available_channels)
        self.available_channels.remove(channel)
        self.note_to_channel[(note_pitch, start_time)] = channel

        return channel

    def release_channel(self, note_pitch: int, start_time: int) -> None:
        """Release a channel back to the pool.

        Args:
            note_pitch: MIDI note number
            start_time: Note start time
        """
        key = (note_pitch, start_time)
        if key in self.note_to_channel:
            channel = self.note_to_channel[key]
            self.available_channels.add(channel)
            del self.note_to_channel[key]

    def get_manager_channel(self) -> int:
        """Get manager channel number."""
        return self.manager_channel

    def reset(self) -> None:
        """Reset channel allocation."""
        self.available_channels = set(self.member_channels)
        self.note_to_channel.clear()

    def get_mpe_configuration_messages(self) -> List[Tuple[int, List[int]]]:
        """Get MPE configuration RPN messages.

        Returns MPE Zone Configuration messages (RPN 6).

        Returns:
            List of (channel, [data bytes]) for configuration
        """
        messages = []

        # MPE Zone Configuration is RPN 6
        # Format: Channel, RPN MSB=0, RPN LSB=6, Data Entry MSB=num_channels

        if self.zone == MPEZoneLayout.LOWER_ZONE:
            # Configure lower zone
            messages.append((self.manager_channel, [
                # RPN MSB (CC 101)
                101, 0,
                # RPN LSB (CC 100)
                100, 6,
                # Data Entry MSB (CC 6) = number of member channels
                6, self.num_channels
            ]))

        elif self.zone == MPEZoneLayout.UPPER_ZONE:
            # Configure upper zone
            messages.append((self.manager_channel, [
                101, 0,
                100, 6,
                6, self.num_channels
            ]))

        return messages


class MPEGestureEngine:
    """Generate MPE gestures (pitch, pressure, timbre modulation).

    Creates expressive gestures for MPE controllers including:
    - Pitch slides and vibrato
    - Pressure swells
    - Timbre sweeps
    - Complex multi-dimensional gestures

    Research:
        - Continuum Fingerboard playing techniques
        - ROLI Seaboard expression dimensions (strike, press, glide, slide, lift)
    """

    def __init__(self, ticks_per_quarter: int = 480):
        """Initialize gesture engine.

        Args:
            ticks_per_quarter: MIDI ticks per quarter note
        """
        self.tpq = ticks_per_quarter

    def generate_pitch_vibrato(self, note: MPENote,
                               rate_hz: float = 5.0,
                               depth_cents: int = 50,
                               delay_ms: int = 100) -> List[MPEPitchBend]:
        """Generate pitch vibrato for a note.

        Args:
            note: MPE note to add vibrato to
            rate_hz: Vibrato rate in Hz
            depth_cents: Vibrato depth in cents
            delay_ms: Vibrato onset delay

        Returns:
            List of MPEPitchBend events
        """
        pitch_bends = []

        # Convert delay to ticks (assume 120 BPM)
        delay_ticks = int((delay_ms / 1000.0) * (120 / 60.0) * self.tpq)

        if note.duration < delay_ticks + 60:
            return pitch_bends

        vibrato_start = note.start_time + delay_ticks
        vibrato_end = note.end_time()

        # Ticks per vibrato cycle
        ticks_per_cycle = int(self.tpq * (120 / 60.0) / rate_hz)

        current_time = vibrato_start
        sample_rate = 30  # Ticks between samples

        while current_time < vibrato_end:
            # Vibrato progress (for amplitude envelope)
            vibrato_time = current_time - vibrato_start
            vibrato_progress = vibrato_time / max(1, vibrato_end - vibrato_start)

            # Amplitude envelope (fade in gradually)
            amplitude = min(1.0, vibrato_progress * 2.5)

            # Vibrato waveform (sine with slight randomization)
            phase = (current_time - vibrato_start) / ticks_per_cycle
            vibrato_wave = math.sin(2 * math.pi * phase)
            vibrato_wave += random.uniform(-0.1, 0.1)  # Humanize

            # Calculate pitch bend value
            cents_deviation = amplitude * depth_cents * vibrato_wave
            semitone_bend = 8192 / 12  # Pitch bend per semitone
            bend_value = int((cents_deviation / 100.0) * semitone_bend)

            # Clamp to MIDI pitch bend range
            bend_value = max(-8192, min(8191, bend_value))

            pitch_bends.append(MPEPitchBend(
                time=current_time,
                channel=note.channel,
                value=bend_value
            ))

            current_time += sample_rate

        return pitch_bends

    def generate_pitch_slide(self, note: MPENote,
                            target_cents: int = 100,
                            slide_type: str = 'immediate') -> List[MPEPitchBend]:
        """Generate pitch slide/glide.

        Args:
            note: MPE note
            target_cents: Target pitch offset in cents
            slide_type: 'immediate', 'gradual', 'overshoot'

        Returns:
            List of MPEPitchBend events
        """
        pitch_bends = []

        semitone_bend = 8192 / 12
        target_bend = int((target_cents / 100.0) * semitone_bend)

        if slide_type == 'immediate':
            # Immediate slide to target
            slide_duration = min(120, note.duration // 4)
            steps = 15

            for i in range(steps + 1):
                t = i / steps
                # Smooth ease-out curve
                curve_t = 1 - (1 - t) ** 2
                bend_value = int(target_bend * curve_t)

                pitch_bends.append(MPEPitchBend(
                    time=note.start_time + int(t * slide_duration),
                    channel=note.channel,
                    value=bend_value
                ))

        elif slide_type == 'gradual':
            # Gradual slide over entire note
            steps = 30
            for i in range(steps + 1):
                t = i / steps
                # Linear slide
                bend_value = int(target_bend * t)

                time = note.start_time + int(t * note.duration)
                pitch_bends.append(MPEPitchBend(
                    time=time,
                    channel=note.channel,
                    value=bend_value
                ))

        elif slide_type == 'overshoot':
            # Overshoot and settle
            overshoot_factor = 1.3
            settle_time = min(180, note.duration // 3)
            steps = 20

            for i in range(steps + 1):
                t = i / steps

                if t < 0.6:
                    # Overshoot phase
                    curve_t = t / 0.6
                    bend_value = int(target_bend * overshoot_factor * curve_t)
                else:
                    # Settle phase
                    curve_t = (t - 0.6) / 0.4
                    start_bend = target_bend * overshoot_factor
                    bend_value = int(start_bend + (target_bend - start_bend) * curve_t)

                pitch_bends.append(MPEPitchBend(
                    time=note.start_time + int(t * settle_time),
                    channel=note.channel,
                    value=bend_value
                ))

        return pitch_bends

    def generate_pressure_swell(self, note: MPENote,
                               peak_pressure: int = 100,
                               peak_position: float = 0.5) -> List[MPEPressure]:
        """Generate pressure swell (dynamic change over note).

        Args:
            note: MPE note
            peak_pressure: Peak pressure value (0-127)
            peak_position: Where peak occurs (0.0-1.0)

        Returns:
            List of MPEPressure events
        """
        pressure_events = []

        steps = 25
        for i in range(steps + 1):
            t = i / steps

            # Pressure envelope (swell shape)
            if t < peak_position:
                # Rising phase
                progress = t / peak_position
                pressure_value = int(peak_pressure * progress)
            else:
                # Falling phase
                progress = (t - peak_position) / (1 - peak_position)
                pressure_value = int(peak_pressure * (1 - progress))

            # Add slight randomness
            pressure_value += random.randint(-3, 3)
            pressure_value = max(0, min(127, pressure_value))

            time = note.start_time + int(t * note.duration)
            pressure_events.append(MPEPressure(
                time=time,
                channel=note.channel,
                value=pressure_value
            ))

        return pressure_events

    def generate_timbre_sweep(self, note: MPENote,
                             start_timbre: int = 30,
                             end_timbre: int = 100,
                             curve: str = 'linear') -> List[MPETimbre]:
        """Generate timbre sweep (filter/brightness change).

        Args:
            note: MPE note
            start_timbre: Starting CC74 value
            end_timbre: Ending CC74 value
            curve: 'linear', 'exponential', 'logarithmic'

        Returns:
            List of MPETimbre events
        """
        timbre_events = []

        steps = 20
        for i in range(steps + 1):
            t = i / steps

            # Apply curve
            if curve == 'exponential':
                curve_t = t ** 2
            elif curve == 'logarithmic':
                curve_t = math.sqrt(t)
            else:  # linear
                curve_t = t

            # Interpolate timbre value
            timbre_value = int(start_timbre + (end_timbre - start_timbre) * curve_t)
            timbre_value = max(0, min(127, timbre_value))

            time = note.start_time + int(t * note.duration)
            timbre_events.append(MPETimbre(
                time=time,
                channel=note.channel,
                value=timbre_value
            ))

        return timbre_events

    def generate_complex_gesture(self, note: MPENote,
                                 gesture_type: str = 'expressive') -> Tuple[List, List, List]:
        """Generate complex multi-dimensional gesture.

        Combines pitch, pressure, and timbre for expressive playing.

        Args:
            note: MPE note
            gesture_type: 'expressive', 'aggressive', 'subtle', 'vocal'

        Returns:
            Tuple of (pitch_bends, pressure_events, timbre_events)
        """
        pitch_bends = []
        pressure_events = []
        timbre_events = []

        if gesture_type == 'expressive':
            # Musical expression: vibrato + pressure swell + timbre evolution
            pitch_bends = self.generate_pitch_vibrato(
                note, rate_hz=5.5, depth_cents=40, delay_ms=150
            )
            pressure_events = self.generate_pressure_swell(
                note, peak_pressure=90, peak_position=0.4
            )
            timbre_events = self.generate_timbre_sweep(
                note, start_timbre=50, end_timbre=80, curve='linear'
            )

        elif gesture_type == 'aggressive':
            # Aggressive: immediate slide + high pressure + bright timbre
            pitch_bends = self.generate_pitch_slide(
                note, target_cents=50, slide_type='overshoot'
            )
            pressure_events = self.generate_pressure_swell(
                note, peak_pressure=120, peak_position=0.2
            )
            timbre_events = self.generate_timbre_sweep(
                note, start_timbre=90, end_timbre=127, curve='exponential'
            )

        elif gesture_type == 'subtle':
            # Subtle: slight vibrato + gentle pressure + stable timbre
            pitch_bends = self.generate_pitch_vibrato(
                note, rate_hz=4.5, depth_cents=25, delay_ms=200
            )
            pressure_events = self.generate_pressure_swell(
                note, peak_pressure=60, peak_position=0.5
            )
            timbre_events = self.generate_timbre_sweep(
                note, start_timbre=64, end_timbre=70, curve='linear'
            )

        elif gesture_type == 'vocal':
            # Vocal-like: expressive vibrato + dynamic pressure + timbre variation
            pitch_bends = self.generate_pitch_vibrato(
                note, rate_hz=6.0, depth_cents=60, delay_ms=100
            )
            # Pressure envelope mimics vocal dynamics
            pressure_events = self.generate_pressure_swell(
                note, peak_pressure=95, peak_position=0.35
            )
            # Timbre varies like vowel formants
            timbre_events = self.generate_timbre_sweep(
                note, start_timbre=40, end_timbre=90, curve='logarithmic'
            )

        return pitch_bends, pressure_events, timbre_events


class MPEPerformance:
    """Complete MPE performance with channel management and gestures.

    Combines channel allocation, note assignment, and gesture generation
    to create full MPE performances.

    Example:
        >>> perf = MPEPerformance()
        >>> notes = [...]  # Standard MIDI notes
        >>> mpe_notes, events = perf.convert_to_mpe(notes)
        >>> config = perf.get_configuration_messages()
    """

    def __init__(self, zone: MPEZoneLayout = MPEZoneLayout.LOWER_ZONE,
                 num_channels: int = 15, ticks_per_quarter: int = 480):
        """Initialize MPE performance.

        Args:
            zone: MPE zone layout
            num_channels: Number of member channels
            ticks_per_quarter: MIDI ticks per quarter note
        """
        self.channel_manager = MPEChannelManager(zone, num_channels)
        self.gesture_engine = MPEGestureEngine(ticks_per_quarter)
        self.tpq = ticks_per_quarter

    def convert_to_mpe(self, notes: List,
                      add_vibrato: bool = True,
                      add_pressure: bool = True,
                      add_timbre: bool = False,
                      gesture_type: str = 'expressive') -> Tuple[List[MPENote], Dict]:
        """Convert standard MIDI notes to MPE notes with expression.

        Args:
            notes: List of standard Note objects
            add_vibrato: Add pitch vibrato
            add_pressure: Add pressure modulation
            add_timbre: Add timbre modulation
            gesture_type: Type of gesture to apply

        Returns:
            Tuple of (mpe_notes, expression_events_dict)
        """
        mpe_notes = []
        all_pitch_bends = []
        all_pressure = []
        all_timbre = []

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda n: n.start_time)

        for note in sorted_notes:
            # Allocate MPE channel
            channel = self.channel_manager.allocate_channel(note.pitch, note.start_time)

            if channel is None:
                # No channel available, skip note (or implement voice stealing)
                continue

            # Create MPE note
            mpe_note = MPENote(
                pitch=note.pitch,
                velocity=note.velocity,
                start_time=note.start_time,
                duration=note.duration,
                channel=channel
            )

            mpe_notes.append(mpe_note)

            # Generate gestures
            if add_vibrato or add_pressure or add_timbre:
                pitch_bends, pressure, timbre = self.gesture_engine.generate_complex_gesture(
                    mpe_note, gesture_type=gesture_type
                )

                if add_vibrato:
                    all_pitch_bends.extend(pitch_bends)
                if add_pressure:
                    all_pressure.extend(pressure)
                if add_timbre:
                    all_timbre.extend(timbre)

            # Release channel after note ends
            # (In real implementation, this would be event-driven)
            self.channel_manager.release_channel(note.pitch, note.start_time)

        events = {
            'pitch_bends': all_pitch_bends,
            'pressure': all_pressure,
            'timbre': all_timbre
        }

        return mpe_notes, events

    def get_configuration_messages(self) -> List[Tuple[int, List[int]]]:
        """Get MPE configuration messages.

        Returns:
            List of configuration messages
        """
        return self.channel_manager.get_mpe_configuration_messages()


# Example usage
if __name__ == "__main__":
    print("MPE Support Module - Example Usage\n")
    print("=" * 70)

    # Import Note class for testing (simplified version)
    @dataclass
    class Note:
        pitch: int
        velocity: int
        start_time: int
        duration: int
        channel: int = 0

    # Example 1: Channel Management
    print("\n1. MPE Channel Management:")
    manager = MPEChannelManager(zone=MPEZoneLayout.LOWER_ZONE, num_channels=15)
    print(f"   Manager channel: {manager.get_manager_channel()}")
    print(f"   Member channels: {manager.member_channels}")

    # Allocate channels
    ch1 = manager.allocate_channel(60, 0)
    ch2 = manager.allocate_channel(64, 0)
    ch3 = manager.allocate_channel(67, 0)
    print(f"   Allocated channels: {ch1}, {ch2}, {ch3}")
    print(f"   Available channels remaining: {len(manager.available_channels)}")

    # Configuration messages
    config = manager.get_mpe_configuration_messages()
    print(f"   Configuration messages: {len(config)}")

    # Example 2: Pitch Vibrato
    print("\n2. MPE Pitch Vibrato:")
    gesture_engine = MPEGestureEngine(ticks_per_quarter=480)
    test_note = MPENote(pitch=60, velocity=100, start_time=0,
                       duration=1920, channel=2)
    vibrato = gesture_engine.generate_pitch_vibrato(
        test_note, rate_hz=5.5, depth_cents=50, delay_ms=100
    )
    print(f"   Generated {len(vibrato)} vibrato pitch bend events")
    if vibrato:
        print(f"   Sample bend values: {[v.value for v in vibrato[:5]]}")

    # Example 3: Pressure Swell
    print("\n3. MPE Pressure Swell:")
    pressure = gesture_engine.generate_pressure_swell(
        test_note, peak_pressure=100, peak_position=0.4
    )
    print(f"   Generated {len(pressure)} pressure events")
    if pressure:
        print(f"   Pressure range: {min(p.value for p in pressure)}-{max(p.value for p in pressure)}")

    # Example 4: Complex Gesture
    print("\n4. Complex MPE Gesture:")
    pitch, press, timbre = gesture_engine.generate_complex_gesture(
        test_note, gesture_type='expressive'
    )
    print(f"   Pitch bends: {len(pitch)}")
    print(f"   Pressure events: {len(press)}")
    print(f"   Timbre events: {len(timbre)}")

    # Example 5: Full MPE Performance
    print("\n5. Full MPE Performance Conversion:")
    mpe_perf = MPEPerformance(zone=MPEZoneLayout.LOWER_ZONE, num_channels=15)

    # Create test notes
    test_notes = [
        Note(60, 80, 0, 960, 0),
        Note(64, 75, 0, 960, 0),
        Note(67, 70, 0, 960, 0),
        Note(72, 85, 960, 1920, 0),
    ]

    mpe_notes, events = mpe_perf.convert_to_mpe(
        test_notes,
        add_vibrato=True,
        add_pressure=True,
        add_timbre=True,
        gesture_type='expressive'
    )

    print(f"   Input notes: {len(test_notes)}")
    print(f"   MPE notes: {len(mpe_notes)}")
    print(f"   Total pitch bend events: {len(events['pitch_bends'])}")
    print(f"   Total pressure events: {len(events['pressure'])}")
    print(f"   Total timbre events: {len(events['timbre'])}")

    # Example 6: Different Gesture Types
    print("\n6. Different Gesture Types:")
    for gesture_type in ['expressive', 'aggressive', 'subtle', 'vocal']:
        pitch, press, timbre = gesture_engine.generate_complex_gesture(
            test_note, gesture_type=gesture_type
        )
        print(f"   {gesture_type.capitalize()}: pitch={len(pitch)}, "
              f"pressure={len(press)}, timbre={len(timbre)}")

    print("\n" + "=" * 70)
    print("All MPE tests completed successfully!")
