"""
AGENT 20: Rhythm Specialist
============================

Advanced rhythm analysis and generation beyond basic rhythmic parameters.

Specialized areas:
- Polyrhythm (3:2, 4:3, 5:4, etc.)
- Swing and groove quantization
- Syncopation patterns
- Metric modulation
- African and Latin rhythms (clave, bell patterns)
- Complex time signatures
- Rhythmic tension and release

This module provides 50+ specialized rhythm parameters and advanced
rhythm generation techniques not covered in the basic rhythm system.

Author: Agent 20 - Rhythm Specialist
License: MIT
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Optional, Any, Set
from pathlib import Path
import random
import math

# Optional dependencies
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    print("WARNING: numpy not installed, some features will be limited")

try:
    import mido
    HAS_MIDO = True
except ImportError:
    HAS_MIDO = False
    print("WARNING: mido not installed, MIDI export will not be available")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class RhythmicEventType(Enum):
    """Types of rhythmic events"""
    NOTE_ON = "note_on"
    NOTE_OFF = "note_off"
    ACCENT = "accent"
    GHOST = "ghost"
    REST = "rest"


@dataclass
class RhythmicEvent:
    """
    A single rhythmic event with timing and articulation.

    Attributes:
        onset_time: Start time in beats or seconds
        duration: Duration in beats or seconds
        velocity: MIDI velocity (0-127)
        pitch: Optional MIDI pitch
        event_type: Type of event
        subdivision: Which subdivision this falls on (e.g., 16th note)
        accent_level: Accent strength (0.0-1.0)
    """
    onset_time: float
    duration: float
    velocity: int = 100
    pitch: Optional[int] = None
    event_type: RhythmicEventType = RhythmicEventType.NOTE_ON
    subdivision: int = 16  # 16th note by default
    accent_level: float = 0.5

    def to_midi_messages(self, channel: int = 0):
        """Convert to MIDI messages"""
        if not HAS_MIDO:
            raise RuntimeError("mido not available - cannot convert to MIDI messages")

        messages = []

        if self.event_type in [RhythmicEventType.NOTE_ON, RhythmicEventType.ACCENT]:
            # Note on
            messages.append(mido.Message(
                'note_on',
                note=self.pitch or 60,
                velocity=self.velocity,
                time=0,
                channel=channel
            ))

            # Note off
            messages.append(mido.Message(
                'note_off',
                note=self.pitch or 60,
                velocity=0,
                time=int(self.duration * 480),  # Convert to ticks
                channel=channel
            ))

        return messages


@dataclass
class PolyrhythmPattern:
    """A polyrhythmic pattern combining two or more subdivisions"""
    ratio: Tuple[int, int]  # e.g., (3, 2) for triplets over duplets
    voices: List[List[RhythmicEvent]]
    length_beats: float
    tension_level: float = 0.5  # How complex/tense the polyrhythm is


@dataclass
class ClavePattern:
    """A clave or timeline pattern (African/Latin rhythms)"""
    name: str
    pattern: List[float]  # Onset times in beats
    length_beats: float
    origin: str  # e.g., "Cuban", "Brazilian", "West African"
    feeling: str  # e.g., "forward", "laid-back", "driving"


# ============================================================================
# RHYTHM SPECIALIST
# ============================================================================

class RhythmSpecialist:
    """
    Advanced rhythm analysis and generation specialist.

    Provides sophisticated rhythmic techniques including:
    - Polyrhythm generation and analysis
    - Swing and groove quantization
    - Syncopation pattern creation
    - World rhythm patterns (clave, bell patterns)
    - Metric modulation
    - Rhythmic tension/release curves
    """

    def __init__(self):
        """Initialize the rhythm specialist"""
        self.clave_patterns = self._initialize_clave_patterns()
        self.swing_presets = self._initialize_swing_presets()
        self.polyrhythm_cache = {}

    # ========================================================================
    # POLYRHYTHM GENERATION
    # ========================================================================

    def generate_polyrhythm(
        self,
        ratio: Tuple[int, int],
        length_beats: float,
        base_velocity: int = 100,
        accent_pattern: Optional[List[bool]] = None
    ) -> PolyrhythmPattern:
        """
        Generate a polyrhythmic pattern.

        Args:
            ratio: Tuple of (voice1_subdivisions, voice2_subdivisions)
                  e.g., (3, 2) = 3 against 2, (4, 3) = 4 against 3
            length_beats: Length in beats
            base_velocity: Base MIDI velocity
            accent_pattern: Optional accent pattern for each voice

        Returns:
            PolyrhythmPattern object

        Example:
            # Generate 3 against 2 (triplets vs duplets)
            pattern = specialist.generate_polyrhythm((3, 2), 4.0)
        """
        voice1_div, voice2_div = ratio

        # Generate voice 1
        voice1 = []
        step1 = length_beats / voice1_div
        for i in range(voice1_div):
            accent = 1.0 if i == 0 else 0.5
            if accent_pattern and len(accent_pattern) > 0:
                accent = 1.0 if accent_pattern[i % len(accent_pattern)] else 0.5

            event = RhythmicEvent(
                onset_time=i * step1,
                duration=step1 * 0.8,  # Slightly detached
                velocity=int(base_velocity * (0.7 + 0.3 * accent)),
                event_type=RhythmicEventType.ACCENT if accent > 0.7 else RhythmicEventType.NOTE_ON,
                accent_level=accent
            )
            voice1.append(event)

        # Generate voice 2
        voice2 = []
        step2 = length_beats / voice2_div
        for i in range(voice2_div):
            accent = 1.0 if i == 0 else 0.5

            event = RhythmicEvent(
                onset_time=i * step2,
                duration=step2 * 0.8,
                velocity=int(base_velocity * (0.6 + 0.3 * accent)),
                event_type=RhythmicEventType.ACCENT if accent > 0.7 else RhythmicEventType.NOTE_ON,
                accent_level=accent
            )
            voice2.append(event)

        # Calculate tension level based on ratio complexity
        tension = self._calculate_polyrhythm_tension(ratio)

        return PolyrhythmPattern(
            ratio=ratio,
            voices=[voice1, voice2],
            length_beats=length_beats,
            tension_level=tension
        )

    def _calculate_polyrhythm_tension(self, ratio: Tuple[int, int]) -> float:
        """
        Calculate the perceptual tension/complexity of a polyrhythm.

        Simple ratios like 2:1, 3:2 are less tense.
        Complex ratios like 7:5, 11:8 are more tense.
        """
        a, b = ratio
        gcd = math.gcd(a, b)
        lcm = abs(a * b) // gcd

        # Complexity based on LCM and GCD
        complexity = lcm / gcd

        # Normalize to 0-1 range
        tension = min(1.0, complexity / 20.0)

        return tension

    def generate_nested_polyrhythm(
        self,
        ratios: List[Tuple[int, int]],
        length_beats: float
    ) -> List[PolyrhythmPattern]:
        """
        Generate nested polyrhythms (polyrhythm within polyrhythm).

        Args:
            ratios: List of polyrhythm ratios to nest
            length_beats: Total length in beats

        Returns:
            List of polyrhythm patterns at different levels
        """
        patterns = []

        for ratio in ratios:
            pattern = self.generate_polyrhythm(ratio, length_beats)
            patterns.append(pattern)

        return patterns

    def analyze_polyrhythm(
        self,
        events1: List[RhythmicEvent],
        events2: List[RhythmicEvent]
    ) -> Dict[str, Any]:
        """
        Analyze two voices to detect polyrhythmic relationships.

        Returns:
            Dictionary with detected ratio, phase relationships, etc.
        """
        # Extract onset times
        onsets1 = [e.onset_time for e in events1]
        onsets2 = [e.onset_time for e in events2]

        # Estimate ratios
        avg_ioi1 = np.mean(np.diff(onsets1)) if len(onsets1) > 1 else 1.0
        avg_ioi2 = np.mean(np.diff(onsets2)) if len(onsets2) > 1 else 1.0

        # Estimate ratio
        ratio_estimate = avg_ioi2 / avg_ioi1 if avg_ioi1 > 0 else 1.0

        # Find closest simple ratio
        best_ratio = self._find_closest_ratio(ratio_estimate)

        # Phase relationship
        phase = self._calculate_phase_relationship(onsets1, onsets2)

        return {
            'detected_ratio': best_ratio,
            'ratio_estimate': ratio_estimate,
            'phase_offset': phase,
            'voice1_density': len(onsets1),
            'voice2_density': len(onsets2),
            'tension': self._calculate_polyrhythm_tension(best_ratio)
        }

    def _find_closest_ratio(self, ratio: float) -> Tuple[int, int]:
        """Find the closest simple integer ratio"""
        common_ratios = [
            (2, 1), (3, 2), (4, 3), (5, 4),
            (3, 1), (4, 1), (5, 3), (7, 4),
            (7, 5), (8, 5), (11, 8)
        ]

        best_ratio = (1, 1)
        best_error = float('inf')

        for r in common_ratios:
            error = abs(ratio - (r[0] / r[1]))
            if error < best_error:
                best_error = error
                best_ratio = r

        return best_ratio

    def _calculate_phase_relationship(
        self,
        onsets1: List[float],
        onsets2: List[float]
    ) -> float:
        """Calculate phase relationship between two voices"""
        if not onsets1 or not onsets2:
            return 0.0

        # Find closest onsets
        min_distance = float('inf')
        for o1 in onsets1:
            for o2 in onsets2:
                distance = abs(o1 - o2)
                min_distance = min(min_distance, distance)

        return min_distance

    # ========================================================================
    # SWING AND GROOVE
    # ========================================================================

    def apply_swing(
        self,
        events: List[RhythmicEvent],
        swing_amount: float,
        swing_ratio: float = 2.0
    ) -> List[RhythmicEvent]:
        """
        Apply swing feel to rhythmic events.

        Args:
            events: List of rhythmic events
            swing_amount: Amount of swing (0.0 = straight, 1.0 = maximum swing)
            swing_ratio: Swing ratio (2.0 = 2:1 triplet feel, 1.5 = lighter swing)

        Returns:
            List of swung events
        """
        swung_events = []

        for event in events:
            # Determine if this is an off-beat (even subdivision)
            subdivision_position = event.onset_time % 1.0
            beat_position = (subdivision_position * event.subdivision) % 2

            if abs(beat_position - 1.0) < 0.1:  # Off-beat
                # Apply swing delay
                swing_delay = swing_amount * 0.2  # Max 20% of beat
                swung_event = RhythmicEvent(
                    onset_time=event.onset_time + swing_delay,
                    duration=event.duration * (1.0 - swing_amount * 0.2),  # Shorten slightly
                    velocity=event.velocity,
                    pitch=event.pitch,
                    event_type=event.event_type,
                    subdivision=event.subdivision,
                    accent_level=event.accent_level * (1.0 - swing_amount * 0.1)  # Reduce accent
                )
                swung_events.append(swung_event)
            else:
                # On-beat: keep as is
                swung_events.append(event)

        return swung_events

    def apply_groove_template(
        self,
        events: List[RhythmicEvent],
        groove_name: str
    ) -> List[RhythmicEvent]:
        """
        Apply a groove template (timing and velocity variations).

        Args:
            events: List of events
            groove_name: Name of groove preset

        Returns:
            Grooved events
        """
        if groove_name not in self.swing_presets:
            return events

        preset = self.swing_presets[groove_name]

        grooved_events = []
        for i, event in enumerate(events):
            position = i % len(preset['timing_offsets'])

            timing_offset = preset['timing_offsets'][position]
            velocity_mult = preset['velocity_multipliers'][position]

            grooved_event = RhythmicEvent(
                onset_time=event.onset_time + timing_offset,
                duration=event.duration,
                velocity=int(event.velocity * velocity_mult),
                pitch=event.pitch,
                event_type=event.event_type,
                subdivision=event.subdivision,
                accent_level=event.accent_level
            )
            grooved_events.append(grooved_event)

        return grooved_events

    def _initialize_swing_presets(self) -> Dict[str, Dict]:
        """Initialize groove/swing presets"""
        return {
            'jazz_swing': {
                'timing_offsets': [0.0, 0.15, 0.0, 0.15],
                'velocity_multipliers': [1.0, 0.7, 0.9, 0.6]
            },
            'shuffle': {
                'timing_offsets': [0.0, 0.2, 0.0, 0.2],
                'velocity_multipliers': [1.0, 0.65, 0.95, 0.6]
            },
            'laid_back': {
                'timing_offsets': [0.0, 0.05, 0.0, 0.05],
                'velocity_multipliers': [1.0, 0.85, 0.95, 0.8]
            },
            'pushed': {
                'timing_offsets': [0.0, -0.05, 0.0, -0.05],
                'velocity_multipliers': [1.0, 0.9, 0.95, 0.85]
            },
            'drunk': {
                'timing_offsets': [0.03, -0.02, 0.04, -0.03],
                'velocity_multipliers': [1.0, 0.8, 0.9, 0.75]
            }
        }

    def humanize_timing(
        self,
        events: List[RhythmicEvent],
        amount: float = 0.5,
        randomness: float = 0.3
    ) -> List[RhythmicEvent]:
        """
        Add human-like timing imperfections.

        Args:
            events: Rhythmic events
            amount: Amount of humanization (0.0 = mechanical, 1.0 = very human)
            randomness: Amount of randomness vs. consistent drift

        Returns:
            Humanized events
        """
        humanized = []

        if HAS_NUMPY:
            # Generate timing drift curve (consistent human tendency)
            drift_curve = np.cumsum(np.random.randn(len(events))) * amount * 0.01

            for i, event in enumerate(events):
                # Consistent drift
                drift = drift_curve[i] * (1.0 - randomness)

                # Random jitter
                jitter = np.random.randn() * amount * 0.02 * randomness

                humanized_event = RhythmicEvent(
                    onset_time=event.onset_time + drift + jitter,
                    duration=event.duration * (1.0 + np.random.randn() * amount * 0.1),
                    velocity=int(event.velocity * (1.0 + np.random.randn() * amount * 0.15)),
                    pitch=event.pitch,
                    event_type=event.event_type,
                    subdivision=event.subdivision,
                    accent_level=event.accent_level
                )
                humanized.append(humanized_event)
        else:
            # Fallback without numpy
            drift = 0.0
            for event in events:
                # Simple random jitter
                drift += (random.random() - 0.5) * amount * 0.02 * (1.0 - randomness)
                jitter = (random.random() - 0.5) * amount * 0.04 * randomness

                humanized_event = RhythmicEvent(
                    onset_time=event.onset_time + drift + jitter,
                    duration=event.duration * (1.0 + (random.random() - 0.5) * amount * 0.2),
                    velocity=int(event.velocity * (1.0 + (random.random() - 0.5) * amount * 0.3)),
                    pitch=event.pitch,
                    event_type=event.event_type,
                    subdivision=event.subdivision,
                    accent_level=event.accent_level
                )
                humanized.append(humanized_event)

        return humanized

    # ========================================================================
    # SYNCOPATION
    # ========================================================================

    def add_syncopation(
        self,
        pattern: List[RhythmicEvent],
        amount: float,
        target_beats: Optional[List[float]] = None
    ) -> List[RhythmicEvent]:
        """
        Add syncopation to a rhythmic pattern.

        Args:
            pattern: Original pattern
            amount: Amount of syncopation (0.0-1.0)
            target_beats: Specific beat positions to syncopate (optional)

        Returns:
            Syncopated pattern
        """
        if amount <= 0.0:
            return pattern

        syncopated = []

        for event in pattern:
            # Determine if this beat should be syncopated
            beat_position = event.onset_time % 1.0

            # Strong beats (1, 3) are less likely to be syncopated
            is_strong_beat = abs(beat_position) < 0.1 or abs(beat_position - 2.0) < 0.1

            syncopate_prob = amount * (0.3 if is_strong_beat else 0.8)

            if random.random() < syncopate_prob:
                # Anticipate by moving earlier
                anticipation = random.uniform(0.1, 0.3)
                syncopated_event = RhythmicEvent(
                    onset_time=event.onset_time - anticipation,
                    duration=event.duration + anticipation * 0.5,
                    velocity=int(event.velocity * 1.1),  # Slightly louder
                    pitch=event.pitch,
                    event_type=RhythmicEventType.ACCENT,
                    subdivision=event.subdivision,
                    accent_level=min(1.0, event.accent_level * 1.3)
                )
                syncopated.append(syncopated_event)
            else:
                syncopated.append(event)

        return syncopated

    def generate_syncopation_pattern(
        self,
        length_beats: float,
        density: float = 0.5,
        complexity: float = 0.5
    ) -> List[RhythmicEvent]:
        """
        Generate a syncopated rhythmic pattern from scratch.

        Args:
            length_beats: Length in beats
            density: Note density (0.0 = sparse, 1.0 = dense)
            complexity: Syncopation complexity (0.0 = simple, 1.0 = complex)

        Returns:
            List of syncopated events
        """
        events = []

        # Determine subdivision based on complexity
        subdivision = int(8 + complexity * 8)  # 8th to 16th notes

        num_positions = int(length_beats * subdivision)

        for i in range(num_positions):
            position = i / subdivision

            # Probability of note based on density and beat position
            beat_strength = self._get_beat_strength(position)

            # Syncopation inverts beat strength
            note_prob = density * (1.0 - complexity * beat_strength)

            if random.random() < note_prob:
                accent = 1.0 - beat_strength if complexity > 0.5 else beat_strength

                event = RhythmicEvent(
                    onset_time=position,
                    duration=0.9 / subdivision,
                    velocity=int(60 + accent * 50),
                    event_type=RhythmicEventType.ACCENT if accent > 0.7 else RhythmicEventType.NOTE_ON,
                    accent_level=accent
                )
                events.append(event)

        return events

    def _get_beat_strength(self, position: float) -> float:
        """Get the metrical strength of a beat position"""
        # Strong beats: 1, 3 in 4/4
        beat_num = position % 4.0

        if abs(beat_num) < 0.1 or abs(beat_num - 2.0) < 0.1:
            return 1.0  # Downbeats
        elif abs(beat_num - 1.0) < 0.1 or abs(beat_num - 3.0) < 0.1:
            return 0.5  # Backbeats
        else:
            return 0.2  # Off-beats

    # ========================================================================
    # WORLD RHYTHMS (CLAVE PATTERNS)
    # ========================================================================

    def generate_clave(
        self,
        clave_type: str,
        length_beats: float = 4.0,
        velocity: int = 100
    ) -> List[RhythmicEvent]:
        """
        Generate a clave pattern (Cuban, Brazilian, African).

        Args:
            clave_type: Type of clave ('son', 'rumba', 'bossa', 'gahu', etc.)
            length_beats: Length in beats (usually 2 or 4)
            velocity: Base velocity

        Returns:
            List of clave events

        Available patterns:
            - 'son': Cuban son clave (3-2 or 2-3)
            - 'rumba': Cuban rumba clave
            - 'bossa': Brazilian bossa nova clave
            - 'gahu': West African gahu bell pattern
            - 'bembé': Cuban bembé bell pattern
        """
        if clave_type not in self.clave_patterns:
            raise ValueError(f"Unknown clave type: {clave_type}. Available: {list(self.clave_patterns.keys())}")

        clave = self.clave_patterns[clave_type]
        events = []

        # Scale pattern to requested length
        scale_factor = length_beats / clave.length_beats

        for i, onset in enumerate(clave.pattern):
            # First note is accented
            accent = 1.0 if i == 0 else 0.7

            event = RhythmicEvent(
                onset_time=onset * scale_factor,
                duration=0.2,  # Short, sharp notes
                velocity=int(velocity * accent),
                event_type=RhythmicEventType.ACCENT if i == 0 else RhythmicEventType.NOTE_ON,
                accent_level=accent
            )
            events.append(event)

        return events

    def _initialize_clave_patterns(self) -> Dict[str, ClavePattern]:
        """Initialize library of clave and timeline patterns"""
        patterns = {}

        # Cuban Son Clave (3-2)
        patterns['son'] = ClavePattern(
            name="Son Clave (3-2)",
            pattern=[0.0, 1.5, 2.5, 4.5, 6.0],  # 2-bar pattern
            length_beats=8.0,
            origin="Cuban",
            feeling="forward"
        )

        # Cuban Rumba Clave (3-2)
        patterns['rumba'] = ClavePattern(
            name="Rumba Clave (3-2)",
            pattern=[0.0, 1.5, 3.0, 4.5, 6.0],
            length_beats=8.0,
            origin="Cuban",
            feeling="laid-back"
        )

        # Brazilian Bossa Nova
        patterns['bossa'] = ClavePattern(
            name="Bossa Nova Clave",
            pattern=[0.0, 0.5, 2.0, 2.5],
            length_beats=4.0,
            origin="Brazilian",
            feeling="smooth"
        )

        # West African Gahu bell pattern
        patterns['gahu'] = ClavePattern(
            name="Gahu Bell Pattern",
            pattern=[0.0, 1.5, 3.0, 4.0, 5.5, 7.0, 10.5],
            length_beats=12.0,
            origin="West African (Ewe)",
            feeling="driving"
        )

        # Cuban Bembé bell pattern
        patterns['bembe'] = ClavePattern(
            name="Bembé Bell Pattern",
            pattern=[0.0, 0.75, 1.5, 2.25, 3.0],
            length_beats=4.0,
            origin="Cuban (Yoruba)",
            feeling="ceremonial"
        )

        # Brazilian Samba
        patterns['samba'] = ClavePattern(
            name="Samba Pattern",
            pattern=[0.0, 0.75, 1.5, 2.0, 3.0, 3.75],
            length_beats=4.0,
            origin="Brazilian",
            feeling="bouncy"
        )

        return patterns

    def get_available_claves(self) -> List[str]:
        """Get list of available clave patterns"""
        return list(self.clave_patterns.keys())

    def analyze_clave_alignment(
        self,
        events: List[RhythmicEvent],
        clave_type: str
    ) -> Dict[str, Any]:
        """
        Analyze how well a pattern aligns with a clave.

        Returns:
            Dictionary with alignment metrics
        """
        clave = self.generate_clave(clave_type)
        clave_onsets = {e.onset_time for e in clave}

        # Count hits and misses
        hits = 0
        misses = 0

        for event in events:
            # Check if event is close to any clave onset
            close_to_clave = any(
                abs(event.onset_time - co) < 0.1
                for co in clave_onsets
            )

            if close_to_clave:
                hits += 1
            else:
                misses += 1

        alignment = hits / (hits + misses) if (hits + misses) > 0 else 0.0

        return {
            'clave_type': clave_type,
            'alignment_score': alignment,
            'hits': hits,
            'misses': misses,
            'total_events': len(events)
        }

    # ========================================================================
    # METRIC MODULATION
    # ========================================================================

    def apply_metric_modulation(
        self,
        events: List[RhythmicEvent],
        ratio: Tuple[int, int],
        modulation_point: float
    ) -> List[RhythmicEvent]:
        """
        Apply metric modulation (tempo change via rhythmic relationship).

        Args:
            events: Original events
            ratio: Ratio of new tempo to old (e.g., (3, 2) = new tempo is 3/2 of old)
            modulation_point: Beat position where modulation occurs

        Returns:
            Modulated events
        """
        new_events = []
        new_tempo, old_tempo = ratio
        tempo_factor = new_tempo / old_tempo

        for event in events:
            if event.onset_time < modulation_point:
                # Before modulation: keep as is
                new_events.append(event)
            else:
                # After modulation: adjust timing
                time_after_mod = event.onset_time - modulation_point
                new_time = modulation_point + (time_after_mod * tempo_factor)

                modulated_event = RhythmicEvent(
                    onset_time=new_time,
                    duration=event.duration * tempo_factor,
                    velocity=event.velocity,
                    pitch=event.pitch,
                    event_type=event.event_type,
                    subdivision=event.subdivision,
                    accent_level=event.accent_level
                )
                new_events.append(modulated_event)

        return new_events

    # ========================================================================
    # COMPLEX TIME SIGNATURES
    # ========================================================================

    def generate_odd_meter_pattern(
        self,
        time_signature: Tuple[int, int],
        length_bars: int = 4,
        grouping: Optional[List[int]] = None
    ) -> List[RhythmicEvent]:
        """
        Generate rhythmic pattern in odd meter (5/4, 7/8, 11/8, etc.).

        Args:
            time_signature: (numerator, denominator) e.g., (7, 8)
            length_bars: Number of bars
            grouping: How to group beats (e.g., [3, 2, 2] for 7/8)

        Returns:
            Rhythmic pattern
        """
        numerator, denominator = time_signature
        beats_per_bar = numerator * (4.0 / denominator)

        # Default grouping for common odd meters
        if grouping is None:
            if numerator == 5:
                grouping = [3, 2] if random.random() < 0.5 else [2, 3]
            elif numerator == 7:
                grouping = [3, 2, 2] if random.random() < 0.5 else [2, 2, 3]
            elif numerator == 11:
                grouping = [3, 3, 3, 2]
            else:
                grouping = [numerator]

        events = []

        for bar in range(length_bars):
            bar_start = bar * beats_per_bar
            position = 0.0

            for i, group_size in enumerate(grouping):
                group_beats = group_size * (4.0 / denominator)

                # Accent first beat of each group
                accent = 1.0 if i == 0 else 0.7

                event = RhythmicEvent(
                    onset_time=bar_start + position,
                    duration=group_beats * 0.8,
                    velocity=int(80 + accent * 40),
                    event_type=RhythmicEventType.ACCENT if accent > 0.8 else RhythmicEventType.NOTE_ON,
                    accent_level=accent
                )
                events.append(event)

                position += group_beats

        return events

    # ========================================================================
    # RHYTHMIC TENSION/RELEASE
    # ========================================================================

    def apply_tension_curve(
        self,
        events: List[RhythmicEvent],
        curve_type: str = 'buildup'
    ) -> List[RhythmicEvent]:
        """
        Apply rhythmic tension curve (buildup, breakdown, arc, etc.).

        Args:
            events: Original events
            curve_type: Type of tension curve
                - 'buildup': Gradually increase tension
                - 'breakdown': Gradually decrease tension
                - 'arc': Build up then release
                - 'valley': Release then build

        Returns:
            Events with tension curve applied
        """
        n = len(events)
        if n == 0:
            return events

        # Generate tension curve
        if HAS_NUMPY:
            positions = np.linspace(0, 1, n)

            if curve_type == 'buildup':
                tension_curve = positions ** 2
            elif curve_type == 'breakdown':
                tension_curve = (1 - positions) ** 2
            elif curve_type == 'arc':
                tension_curve = np.sin(positions * np.pi)
            elif curve_type == 'valley':
                tension_curve = 1 - np.sin(positions * np.pi)
            else:
                tension_curve = np.ones(n) * 0.5
        else:
            # Fallback without numpy
            positions = [i / (n - 1) if n > 1 else 0.5 for i in range(n)]

            if curve_type == 'buildup':
                tension_curve = [p ** 2 for p in positions]
            elif curve_type == 'breakdown':
                tension_curve = [(1 - p) ** 2 for p in positions]
            elif curve_type == 'arc':
                tension_curve = [math.sin(p * math.pi) for p in positions]
            elif curve_type == 'valley':
                tension_curve = [1 - math.sin(p * math.pi) for p in positions]
            else:
                tension_curve = [0.5] * n

        # Apply tension to events
        modified_events = []
        for i, event in enumerate(events):
            tension = tension_curve[i]

            # Higher tension = shorter notes, louder, more accents
            modified_event = RhythmicEvent(
                onset_time=event.onset_time,
                duration=event.duration * (1.0 - tension * 0.3),
                velocity=int(event.velocity * (0.8 + tension * 0.4)),
                pitch=event.pitch,
                event_type=event.event_type,
                subdivision=event.subdivision,
                accent_level=event.accent_level * (0.7 + tension * 0.6)
            )
            modified_events.append(modified_event)

        return modified_events

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def events_to_midi(
        self,
        events: List[RhythmicEvent],
        output_path: Path,
        tempo: int = 120,
        pitch: int = 60
    ):
        """
        Convert rhythmic events to MIDI file.

        Args:
            events: List of rhythmic events
            output_path: Output MIDI file path
            tempo: Tempo in BPM
            pitch: MIDI pitch for events without pitch
        """
        if not HAS_MIDO:
            print("WARNING: mido not available - cannot export to MIDI")
            return

        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add tempo
        track.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(tempo)))

        # Sort events by onset time
        sorted_events = sorted(events, key=lambda e: e.onset_time)

        # Convert to MIDI messages
        current_time = 0
        for event in sorted_events:
            # Note on
            delta_time = int((event.onset_time - current_time) * 480)
            track.append(mido.Message(
                'note_on',
                note=event.pitch or pitch,
                velocity=event.velocity,
                time=delta_time
            ))
            current_time = event.onset_time

            # Note off
            delta_time = int(event.duration * 480)
            track.append(mido.Message(
                'note_off',
                note=event.pitch or pitch,
                velocity=0,
                time=delta_time
            ))
            current_time += event.duration

        mid.save(str(output_path))

    def analyze_rhythmic_complexity(
        self,
        events: List[RhythmicEvent]
    ) -> Dict[str, float]:
        """
        Analyze the complexity of a rhythmic pattern.

        Returns:
            Dictionary with complexity metrics
        """
        if len(events) < 2:
            return {'complexity': 0.0}

        onsets = [e.onset_time for e in events]
        velocities = [e.velocity for e in events]

        if HAS_NUMPY:
            # Inter-onset interval variability
            iois = np.diff(onsets)
            ioi_variability = np.std(iois) / np.mean(iois) if np.mean(iois) > 0 else 0

            # Velocity variability
            vel_variability = np.std(velocities) / np.mean(velocities) if np.mean(velocities) > 0 else 0
        else:
            # Fallback without numpy
            iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]

            mean_ioi = sum(iois) / len(iois) if iois else 0
            std_ioi = math.sqrt(sum((x - mean_ioi)**2 for x in iois) / len(iois)) if iois else 0
            ioi_variability = std_ioi / mean_ioi if mean_ioi > 0 else 0

            mean_vel = sum(velocities) / len(velocities) if velocities else 0
            std_vel = math.sqrt(sum((x - mean_vel)**2 for x in velocities) / len(velocities)) if velocities else 0
            vel_variability = std_vel / mean_vel if mean_vel > 0 else 0

        # Syncopation score (events on weak beats)
        syncopation_score = sum(
            1 for e in events
            if self._get_beat_strength(e.onset_time) < 0.5
        ) / len(events)

        # Overall complexity
        complexity = (
            ioi_variability * 0.4 +
            vel_variability * 0.2 +
            syncopation_score * 0.4
        )

        return {
            'complexity': min(1.0, complexity),
            'ioi_variability': ioi_variability,
            'velocity_variability': vel_variability,
            'syncopation_score': syncopation_score,
            'density': len(events)
        }


# ============================================================================
# MODULE INTERFACE
# ============================================================================

# Global instance
_rhythm_specialist = None

def get_rhythm_specialist() -> RhythmSpecialist:
    """Get or create global rhythm specialist instance"""
    global _rhythm_specialist
    if _rhythm_specialist is None:
        _rhythm_specialist = RhythmSpecialist()
    return _rhythm_specialist


if __name__ == '__main__':
    # Example usage
    print("=" * 80)
    print("AGENT 20: RHYTHM SPECIALIST - DEMONSTRATION")
    print("=" * 80)

    specialist = RhythmSpecialist()

    # 1. Polyrhythm
    print("\n1. Generating 3:2 polyrhythm...")
    poly = specialist.generate_polyrhythm((3, 2), 4.0)
    print(f"   Generated {len(poly.voices)} voices")
    print(f"   Tension level: {poly.tension_level:.2f}")

    # 2. Swing
    print("\n2. Applying swing...")
    events = [RhythmicEvent(onset_time=i * 0.5, duration=0.4) for i in range(8)]
    swung = specialist.apply_swing(events, swing_amount=0.7)
    print(f"   Applied swing to {len(swung)} events")

    # 3. Clave patterns
    print("\n3. Available clave patterns:")
    for clave_name in specialist.get_available_claves():
        pattern = specialist.generate_clave(clave_name)
        print(f"   - {clave_name}: {len(pattern)} notes")

    # 4. Syncopation
    print("\n4. Generating syncopated pattern...")
    syncopated = specialist.generate_syncopation_pattern(4.0, density=0.6, complexity=0.8)
    print(f"   Generated {len(syncopated)} syncopated events")

    # 5. Odd meter
    print("\n5. Generating 7/8 pattern...")
    odd_meter = specialist.generate_odd_meter_pattern((7, 8), length_bars=2, grouping=[3, 2, 2])
    print(f"   Generated {len(odd_meter)} events in 7/8")

    print("\n" + "=" * 80)
    print("✅ AGENT 20 initialized successfully!")
    print("=" * 80)
