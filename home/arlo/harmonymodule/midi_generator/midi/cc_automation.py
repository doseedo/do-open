"""
MIDI CC Automation Engine

Comprehensive MIDI Continuous Controller automation system for realistic expression,
dynamics, and modulation. Supports all standard MIDI CC messages with sophisticated
curve generation and phrase shaping.

Research:
    - MIDI 1.0 Specification (MIDI Manufacturers Association)
    - DAW automation curves (Logic Pro, Ableton Live, Pro Tools)
    - Performance studies: Repp, B. H. (1990). "Patterns of expressive timing"
    - Gabrielsson, A. (2003). "Music Performance Research at the Millennium"

Author: AGENT 6 - MIDI Expression & Performance
"""

from typing import List, Tuple, Optional, Callable, Dict, Union
from dataclasses import dataclass
from enum import Enum
import math
import random


class CCType(Enum):
    """Standard MIDI Continuous Controller types."""
    BANK_SELECT = 0
    MODULATION = 1
    BREATH = 2
    FOOT = 4
    PORTAMENTO_TIME = 5
    DATA_ENTRY = 6
    VOLUME = 7
    BALANCE = 8
    PAN = 10
    EXPRESSION = 11
    EFFECT_1 = 12
    EFFECT_2 = 13
    DAMPER_PEDAL = 64
    PORTAMENTO = 65
    SOSTENUTO = 66
    SOFT_PEDAL = 67
    LEGATO = 68
    HOLD_2 = 69
    SOUND_VARIATION = 70
    TIMBRE = 71
    RELEASE_TIME = 72
    ATTACK_TIME = 73
    BRIGHTNESS = 74
    SOUND_CONTROL_6 = 75
    SOUND_CONTROL_7 = 76
    SOUND_CONTROL_8 = 77
    SOUND_CONTROL_9 = 78
    SOUND_CONTROL_10 = 79
    REVERB = 91
    CHORUS = 93
    DELAY = 94
    PHASER = 95


class CurveType(Enum):
    """Automation curve interpolation types."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    LOGARITHMIC = "logarithmic"
    SINE = "sine"
    COSINE = "cosine"
    S_CURVE = "s_curve"
    STEP = "step"
    RANDOM_WALK = "random_walk"
    CUBIC = "cubic"
    PARABOLIC = "parabolic"


@dataclass
class CCEvent:
    """Single MIDI CC event.

    Attributes:
        time: Time in ticks (MIDI clock pulses)
        cc_number: Controller number (0-127)
        value: Controller value (0-127)
        channel: MIDI channel (0-15)
    """
    time: int
    cc_number: int
    value: int
    channel: int = 0

    def __post_init__(self):
        """Validate CC event parameters."""
        if not 0 <= self.cc_number <= 127:
            raise ValueError(f"CC number must be 0-127, got {self.cc_number}")
        if not 0 <= self.value <= 127:
            raise ValueError(f"CC value must be 0-127, got {self.value}")
        if not 0 <= self.channel <= 15:
            raise ValueError(f"Channel must be 0-15, got {self.channel}")
        if self.time < 0:
            raise ValueError(f"Time must be non-negative, got {self.time}")


@dataclass
class AutomationPoint:
    """Control point for automation curve.

    Attributes:
        time: Time position in ticks
        value: Value at this point (0-127)
        curve: Curve type to next point
    """
    time: float
    value: float
    curve: CurveType = CurveType.LINEAR


class AutomationCurve:
    """Generate smooth automation curves between control points.

    Creates interpolated CC values between automation points using various
    curve types for natural-sounding parameter changes.

    Args:
        cc_number: MIDI CC number to automate
        channel: MIDI channel (0-15)
        resolution: Time resolution in ticks per curve sample

    Example:
        >>> curve = AutomationCurve(cc_number=7, channel=0)  # Volume
        >>> curve.add_point(0, 64, CurveType.LINEAR)
        >>> curve.add_point(1920, 127, CurveType.EXPONENTIAL)
        >>> events = curve.generate(ticks_per_quarter=480)
    """

    def __init__(self, cc_number: int, channel: int = 0, resolution: int = 30):
        """Initialize automation curve.

        Args:
            cc_number: MIDI CC number (0-127)
            channel: MIDI channel (0-15)
            resolution: Ticks between generated events (lower = smoother)
        """
        self.cc_number = cc_number
        self.channel = channel
        self.resolution = resolution
        self.points: List[AutomationPoint] = []

    def add_point(self, time: float, value: float,
                  curve: CurveType = CurveType.LINEAR) -> None:
        """Add control point to automation curve.

        Args:
            time: Time in ticks
            value: CC value (0-127)
            curve: Curve type to next point
        """
        if not 0 <= value <= 127:
            raise ValueError(f"Value must be 0-127, got {value}")

        self.points.append(AutomationPoint(time, value, curve))
        self.points.sort(key=lambda p: p.time)

    def clear(self) -> None:
        """Remove all automation points."""
        self.points.clear()

    def _interpolate_linear(self, t: float, v1: float, v2: float) -> float:
        """Linear interpolation."""
        return v1 + (v2 - v1) * t

    def _interpolate_exponential(self, t: float, v1: float, v2: float,
                                 exponent: float = 2.0) -> float:
        """Exponential interpolation (easing in)."""
        return v1 + (v2 - v1) * (t ** exponent)

    def _interpolate_logarithmic(self, t: float, v1: float, v2: float) -> float:
        """Logarithmic interpolation (easing out)."""
        if t == 0:
            return v1
        return v1 + (v2 - v1) * math.log(1 + t * 9) / math.log(10)

    def _interpolate_sine(self, t: float, v1: float, v2: float) -> float:
        """Sine wave interpolation (smooth S-curve)."""
        sine_t = (1 - math.cos(t * math.pi)) / 2
        return v1 + (v2 - v1) * sine_t

    def _interpolate_cosine(self, t: float, v1: float, v2: float) -> float:
        """Cosine wave interpolation."""
        cos_t = (1 - math.cos(t * math.pi)) / 2
        return v1 + (v2 - v1) * cos_t

    def _interpolate_s_curve(self, t: float, v1: float, v2: float) -> float:
        """S-curve (ease in and out)."""
        s_t = 3 * t**2 - 2 * t**3
        return v1 + (v2 - v1) * s_t

    def _interpolate_cubic(self, t: float, v1: float, v2: float) -> float:
        """Cubic Bezier-style interpolation."""
        cubic_t = t * t * (3 - 2 * t)
        return v1 + (v2 - v1) * cubic_t

    def _interpolate_parabolic(self, t: float, v1: float, v2: float) -> float:
        """Parabolic interpolation (ease out)."""
        para_t = 1 - (1 - t) ** 2
        return v1 + (v2 - v1) * para_t

    def _interpolate(self, time: float, p1: AutomationPoint,
                    p2: AutomationPoint) -> float:
        """Interpolate value at time between two points.

        Args:
            time: Current time in ticks
            p1: Start point
            p2: End point

        Returns:
            Interpolated value (0-127)
        """
        if time <= p1.time:
            return p1.value
        if time >= p2.time:
            return p2.value

        # Normalize time to 0-1
        t = (time - p1.time) / (p2.time - p1.time)

        # Apply curve type
        if p1.curve == CurveType.LINEAR:
            return self._interpolate_linear(t, p1.value, p2.value)
        elif p1.curve == CurveType.EXPONENTIAL:
            return self._interpolate_exponential(t, p1.value, p2.value)
        elif p1.curve == CurveType.LOGARITHMIC:
            return self._interpolate_logarithmic(t, p1.value, p2.value)
        elif p1.curve == CurveType.SINE:
            return self._interpolate_sine(t, p1.value, p2.value)
        elif p1.curve == CurveType.COSINE:
            return self._interpolate_cosine(t, p1.value, p2.value)
        elif p1.curve == CurveType.S_CURVE:
            return self._interpolate_s_curve(t, p1.value, p2.value)
        elif p1.curve == CurveType.CUBIC:
            return self._interpolate_cubic(t, p1.value, p2.value)
        elif p1.curve == CurveType.PARABOLIC:
            return self._interpolate_parabolic(t, p1.value, p2.value)
        elif p1.curve == CurveType.STEP:
            return p1.value
        elif p1.curve == CurveType.RANDOM_WALK:
            # Random walk with drift toward target
            drift = (p2.value - p1.value) * t
            noise = random.uniform(-5, 5)
            return max(0, min(127, p1.value + drift + noise))
        else:
            return self._interpolate_linear(t, p1.value, p2.value)

    def generate(self, start_time: int = 0,
                end_time: Optional[int] = None) -> List[CCEvent]:
        """Generate CC events from automation curve.

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks (None = use last point)

        Returns:
            List of CCEvent objects
        """
        if len(self.points) < 2:
            raise ValueError("Need at least 2 points to generate curve")

        if end_time is None:
            end_time = int(self.points[-1].time)

        events = []
        last_value = None

        for i in range(len(self.points) - 1):
            p1 = self.points[i]
            p2 = self.points[i + 1]

            # Generate events from p1 to p2
            current_time = max(start_time, int(p1.time))
            segment_end = min(end_time, int(p2.time))

            while current_time <= segment_end:
                value = self._interpolate(current_time, p1, p2)
                value_int = max(0, min(127, int(round(value))))

                # Only add if value changed (reduce redundant events)
                if value_int != last_value:
                    events.append(CCEvent(
                        time=current_time,
                        cc_number=self.cc_number,
                        value=value_int,
                        channel=self.channel
                    ))
                    last_value = value_int

                current_time += self.resolution

        return events


class PhraseShaper:
    """Automatic phrase shaping for musical expression.

    Creates natural dynamic arcs, crescendos, decrescendos, and breath marks
    to make MIDI performances more musical and human.

    Research:
        - Sundberg, J. (1988). "Computer synthesis of music performance"
        - Todd, N. P. (1992). "The dynamics of dynamics: A model of musical expression"
    """

    @staticmethod
    def create_crescendo(start_time: int, end_time: int,
                        start_value: int = 64, end_value: int = 127,
                        cc_number: int = 11, channel: int = 0,
                        curve: CurveType = CurveType.EXPONENTIAL) -> AutomationCurve:
        """Create crescendo (gradual increase).

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks
            start_value: Starting CC value (0-127)
            end_value: Ending CC value (0-127)
            cc_number: CC number to automate
            channel: MIDI channel
            curve: Curve type for crescendo

        Returns:
            AutomationCurve configured for crescendo
        """
        auto = AutomationCurve(cc_number, channel)
        auto.add_point(start_time, start_value, curve)
        auto.add_point(end_time, end_value, CurveType.LINEAR)
        return auto

    @staticmethod
    def create_decrescendo(start_time: int, end_time: int,
                          start_value: int = 127, end_value: int = 64,
                          cc_number: int = 11, channel: int = 0,
                          curve: CurveType = CurveType.LOGARITHMIC) -> AutomationCurve:
        """Create decrescendo (gradual decrease).

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks
            start_value: Starting CC value (0-127)
            end_value: Ending CC value (0-127)
            cc_number: CC number to automate
            channel: MIDI channel
            curve: Curve type for decrescendo

        Returns:
            AutomationCurve configured for decrescendo
        """
        auto = AutomationCurve(cc_number, channel)
        auto.add_point(start_time, start_value, curve)
        auto.add_point(end_time, end_value, CurveType.LINEAR)
        return auto

    @staticmethod
    def create_dynamic_arc(start_time: int, peak_time: int, end_time: int,
                          start_value: int = 80, peak_value: int = 110,
                          end_value: int = 75, cc_number: int = 11,
                          channel: int = 0) -> AutomationCurve:
        """Create dynamic arc (crescendo then decrescendo).

        Common in musical phrasing where a phrase builds to a climax
        and then relaxes.

        Args:
            start_time: Phrase start in ticks
            peak_time: Peak/climax time in ticks
            end_time: Phrase end in ticks
            start_value: Starting dynamic level
            peak_value: Peak dynamic level
            end_value: Ending dynamic level
            cc_number: CC number to automate
            channel: MIDI channel

        Returns:
            AutomationCurve with dynamic arc shape
        """
        auto = AutomationCurve(cc_number, channel)
        auto.add_point(start_time, start_value, CurveType.EXPONENTIAL)
        auto.add_point(peak_time, peak_value, CurveType.LOGARITHMIC)
        auto.add_point(end_time, end_value, CurveType.LINEAR)
        return auto

    @staticmethod
    def create_swell(start_time: int, end_time: int,
                    min_value: int = 20, max_value: int = 110,
                    cc_number: int = 11, channel: int = 0) -> AutomationCurve:
        """Create swell (< > dynamic shape).

        Symmetrical crescendo-decrescendo, often used in string and
        wind instrument expressions.

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks
            min_value: Minimum dynamic level
            max_value: Maximum dynamic level
            cc_number: CC number to automate
            channel: MIDI channel

        Returns:
            AutomationCurve with swell shape
        """
        duration = end_time - start_time
        mid_time = start_time + duration // 2

        auto = AutomationCurve(cc_number, channel)
        auto.add_point(start_time, min_value, CurveType.S_CURVE)
        auto.add_point(mid_time, max_value, CurveType.S_CURVE)
        auto.add_point(end_time, min_value, CurveType.LINEAR)
        return auto

    @staticmethod
    def create_breath_mark(time: int, duration: int = 120,
                          depth: int = 30, cc_number: int = 11,
                          channel: int = 0) -> AutomationCurve:
        """Create breath mark (quick dynamic dip).

        Simulates a breath or phrase separation in wind/vocal music.

        Args:
            time: Time of breath mark in ticks
            duration: Duration of breath in ticks
            depth: How much to reduce volume (subtracted from current)
            cc_number: CC number to automate
            channel: MIDI channel

        Returns:
            AutomationCurve for breath mark
        """
        auto = AutomationCurve(cc_number, channel)
        # Assume current value of 100
        current = 100
        auto.add_point(time, current, CurveType.LINEAR)
        auto.add_point(time + duration // 2, current - depth, CurveType.LINEAR)
        auto.add_point(time + duration, current, CurveType.LINEAR)
        return auto


class LFOModulator:
    """Low-Frequency Oscillator for cyclic CC modulation.

    Creates periodic modulation effects like vibrato, tremolo, auto-pan,
    and filter sweeps using various waveform shapes.

    Args:
        cc_number: MIDI CC to modulate
        channel: MIDI channel
        rate_hz: Modulation rate in Hz
        depth: Modulation depth (0-127)
        center: Center value around which to modulate
        waveform: Waveform shape ('sine', 'triangle', 'square', 'sawtooth')
        ticks_per_quarter: MIDI ticks per quarter note (for time conversion)
    """

    def __init__(self, cc_number: int, channel: int = 0,
                 rate_hz: float = 5.0, depth: int = 20,
                 center: int = 64, waveform: str = 'sine',
                 ticks_per_quarter: int = 480):
        self.cc_number = cc_number
        self.channel = channel
        self.rate_hz = rate_hz
        self.depth = depth
        self.center = center
        self.waveform = waveform
        self.ticks_per_quarter = ticks_per_quarter
        self.phase = 0.0

    def _sine_wave(self, phase: float) -> float:
        """Generate sine wave value."""
        return math.sin(phase * 2 * math.pi)

    def _triangle_wave(self, phase: float) -> float:
        """Generate triangle wave value."""
        return 2 * abs(2 * (phase - math.floor(phase + 0.5))) - 1

    def _square_wave(self, phase: float) -> float:
        """Generate square wave value."""
        return 1.0 if (phase % 1.0) < 0.5 else -1.0

    def _sawtooth_wave(self, phase: float) -> float:
        """Generate sawtooth wave value."""
        return 2 * (phase - math.floor(phase + 0.5))

    def generate(self, start_time: int, end_time: int,
                resolution: int = 30) -> List[CCEvent]:
        """Generate LFO modulation events.

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks
            resolution: Time between events in ticks

        Returns:
            List of CCEvent objects
        """
        events = []

        # Convert rate from Hz to cycles per tick
        # Assuming 120 BPM, 480 TPQN: 1 quarter = 480 ticks = 0.5 seconds
        # ticks_per_second = (BPM / 60) * ticks_per_quarter
        # For now, assume 120 BPM
        bpm = 120
        ticks_per_second = (bpm / 60.0) * self.ticks_per_quarter
        cycles_per_tick = self.rate_hz / ticks_per_second

        current_time = start_time
        phase = self.phase

        while current_time <= end_time:
            # Get waveform value (-1 to 1)
            if self.waveform == 'sine':
                wave_value = self._sine_wave(phase)
            elif self.waveform == 'triangle':
                wave_value = self._triangle_wave(phase)
            elif self.waveform == 'square':
                wave_value = self._square_wave(phase)
            elif self.waveform == 'sawtooth':
                wave_value = self._sawtooth_wave(phase)
            else:
                wave_value = self._sine_wave(phase)

            # Scale to CC value
            value = self.center + int(wave_value * self.depth)
            value = max(0, min(127, value))

            events.append(CCEvent(
                time=current_time,
                cc_number=self.cc_number,
                value=value,
                channel=self.channel
            ))

            current_time += resolution
            phase += cycles_per_tick * resolution

        return events


class CCAutomationEngine:
    """Main automation engine for managing multiple CC curves.

    Coordinates multiple automation curves, LFOs, and phrase shapers
    to create complex, layered expressions.

    Example:
        >>> engine = CCAutomationEngine()
        >>> # Add volume crescendo
        >>> vol_curve = PhraseShaper.create_crescendo(0, 1920, 80, 120)
        >>> engine.add_curve('volume', vol_curve)
        >>> # Add vibrato
        >>> vibrato = LFOModulator(cc_number=1, rate_hz=6.0, depth=15)
        >>> engine.add_lfo('vibrato', vibrato)
        >>> # Generate all events
        >>> events = engine.generate_all(0, 1920)
    """

    def __init__(self):
        self.curves: Dict[str, AutomationCurve] = {}
        self.lfos: Dict[str, LFOModulator] = {}

    def add_curve(self, name: str, curve: AutomationCurve) -> None:
        """Add named automation curve.

        Args:
            name: Identifier for this curve
            curve: AutomationCurve object
        """
        self.curves[name] = curve

    def add_lfo(self, name: str, lfo: LFOModulator) -> None:
        """Add named LFO modulator.

        Args:
            name: Identifier for this LFO
            lfo: LFOModulator object
        """
        self.lfos[name] = lfo

    def remove_curve(self, name: str) -> None:
        """Remove automation curve by name."""
        if name in self.curves:
            del self.curves[name]

    def remove_lfo(self, name: str) -> None:
        """Remove LFO modulator by name."""
        if name in self.lfos:
            del self.lfos[name]

    def generate_all(self, start_time: int, end_time: int) -> List[CCEvent]:
        """Generate all automation events from all curves and LFOs.

        Args:
            start_time: Start time in ticks
            end_time: End time in ticks

        Returns:
            Combined list of all CCEvents, sorted by time
        """
        all_events = []

        # Generate from curves
        for curve in self.curves.values():
            try:
                events = curve.generate(start_time, end_time)
                all_events.extend(events)
            except ValueError:
                # Skip curves without enough points
                pass

        # Generate from LFOs
        for lfo in self.lfos.values():
            events = lfo.generate(start_time, end_time)
            all_events.extend(events)

        # Sort by time
        all_events.sort(key=lambda e: (e.time, e.channel, e.cc_number))

        return all_events

    def clear_all(self) -> None:
        """Remove all curves and LFOs."""
        self.curves.clear()
        self.lfos.clear()


def create_volume_automation(start_time: int, end_time: int,
                             dynamics: str = 'mf',
                             shape: str = 'arc',
                             channel: int = 0) -> AutomationCurve:
    """Convenience function to create volume automation.

    Args:
        start_time: Start time in ticks
        end_time: End time in ticks
        dynamics: Dynamic marking ('pp', 'p', 'mp', 'mf', 'f', 'ff')
        shape: Shape of curve ('flat', 'crescendo', 'decrescendo', 'arc', 'swell')
        channel: MIDI channel

    Returns:
        AutomationCurve for volume (CC 7)
    """
    # Dynamic level mappings
    dynamic_values = {
        'ppp': 20, 'pp': 35, 'p': 50, 'mp': 65,
        'mf': 80, 'f': 95, 'ff': 110, 'fff': 127
    }

    base_value = dynamic_values.get(dynamics, 80)

    if shape == 'flat':
        curve = AutomationCurve(CCType.VOLUME.value, channel)
        curve.add_point(start_time, base_value)
        curve.add_point(end_time, base_value)
    elif shape == 'crescendo':
        curve = PhraseShaper.create_crescendo(
            start_time, end_time, base_value, min(127, base_value + 30),
            CCType.VOLUME.value, channel
        )
    elif shape == 'decrescendo':
        curve = PhraseShaper.create_decrescendo(
            start_time, end_time, base_value, max(20, base_value - 30),
            CCType.VOLUME.value, channel
        )
    elif shape == 'arc':
        mid_time = start_time + (end_time - start_time) // 2
        curve = PhraseShaper.create_dynamic_arc(
            start_time, mid_time, end_time,
            base_value, min(127, base_value + 25), base_value,
            CCType.VOLUME.value, channel
        )
    elif shape == 'swell':
        curve = PhraseShaper.create_swell(
            start_time, end_time, max(20, base_value - 20),
            min(127, base_value + 20), CCType.VOLUME.value, channel
        )
    else:
        curve = AutomationCurve(CCType.VOLUME.value, channel)
        curve.add_point(start_time, base_value)
        curve.add_point(end_time, base_value)

    return curve


# Example usage and testing
if __name__ == "__main__":
    print("MIDI CC Automation Engine - Example Usage\n")
    print("=" * 60)

    # Example 1: Simple crescendo
    print("\n1. Creating crescendo automation:")
    crescendo = PhraseShaper.create_crescendo(
        start_time=0,
        end_time=1920,  # 4 beats at 480 TPQN
        start_value=60,
        end_value=120,
        cc_number=11  # Expression
    )
    events = crescendo.generate()
    print(f"   Generated {len(events)} CC events")
    print(f"   First event: time={events[0].time}, value={events[0].value}")
    print(f"   Last event: time={events[-1].time}, value={events[-1].value}")

    # Example 2: LFO modulation (vibrato)
    print("\n2. Creating LFO vibrato:")
    vibrato = LFOModulator(
        cc_number=1,  # Modulation wheel
        rate_hz=6.0,  # 6 Hz vibrato
        depth=20,
        center=64,
        waveform='sine'
    )
    lfo_events = vibrato.generate(0, 1920)
    print(f"   Generated {len(lfo_events)} LFO events")
    print(f"   Rate: 6 Hz, Depth: ±20, Center: 64")

    # Example 3: Complex automation engine
    print("\n3. Using CCAutomationEngine:")
    engine = CCAutomationEngine()

    # Add volume arc
    vol_arc = PhraseShaper.create_dynamic_arc(
        start_time=0,
        peak_time=960,
        end_time=1920,
        start_value=70,
        peak_value=100,
        end_value=65,
        cc_number=7  # Volume
    )
    engine.add_curve('volume', vol_arc)

    # Add pan sweep
    pan_curve = AutomationCurve(cc_number=10, channel=0)  # Pan
    pan_curve.add_point(0, 0, CurveType.SINE)  # Full left
    pan_curve.add_point(1920, 127, CurveType.LINEAR)  # Full right
    engine.add_curve('pan', pan_curve)

    # Generate all
    all_events = engine.generate_all(0, 1920)
    print(f"   Total events from engine: {len(all_events)}")

    # Count by CC type
    cc_counts = {}
    for event in all_events:
        cc_counts[event.cc_number] = cc_counts.get(event.cc_number, 0) + 1
    print(f"   CC breakdown: {cc_counts}")

    # Example 4: Phrase shaping
    print("\n4. Phrase shaping examples:")
    swell = PhraseShaper.create_swell(0, 1920, 40, 100)
    print(f"   Swell: {len(swell.generate())} events")

    breath = PhraseShaper.create_breath_mark(960, duration=120, depth=25)
    print(f"   Breath mark: {len(breath.generate())} events")

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
