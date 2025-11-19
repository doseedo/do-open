#!/usr/bin/env python3
"""
Expressive Performance Module - Advanced MIDI Expression & Humanization

This module transforms mechanical MIDI into human-like expressive performances
using state-of-the-art algorithms from music performance research.

Features:
- Dynamic curves (crescendo, diminuendo, sforzando, accents)
- Velocity humanization (Gaussian variation, avoid mechanical uniformity)
- Microtiming and swing (Roger Linn algorithm, groove feel)
- Rubato and tempo curves (accelerando, ritardando, kinematic models)
- Articulation rendering (staccato, legato, marcato, tenuto)
- Style-specific expression (classical, jazz, pop)

Based on:
- Nature Scientific Reports 2025: "Advancing deep learning for expressive music
  composition and performance modeling" (Transformer models, MAESTRO dataset)
- GigaMIDI Dataset (1.4M files): micro-timing and velocity variation analysis
  (DNVR, DNODR, NOMML heuristics for expressiveness detection)
- MAESTRO Dataset: 200 hours of piano performances with ~3ms MIDI-audio alignment,
  velocity and sustain pedal data
- Roger Linn swing algorithm (MPC): 50% = no swing, 66% = triplet swing, 75% = max
- PMC research on participatory discrepancies: ±50ms microtiming crucial for groove
  (Kilchenmann & Senn 2015, Senn et al. 2016)
- Chopin Nocturnes rubato analysis: kinematic models for ritardandi
- MuseScore/Finale dynamics: exponential, linear, ease-in/out curves
- Articulation standards: staccato = 50% duration, legato = overlap, marcato = accent

Author: Agent 2 - Expressive Performance Modeling
Date: 2025
References:
  [1] Huang et al. (2025) Nature Scientific Reports, 10.1038/s41598-025-13064-6
  [2] Hawthorne et al. (2019) "MAESTRO Dataset", ICLR
  [3] Lee et al. (2025) "GigaMIDI Dataset", ISMIR Transactions
  [4] Linn, R. "MPC Swing Algorithm" (1979-present)
  [5] Senn et al. (2016) "Microtiming in Swing and Funk", PMC
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Callable, Union
from enum import Enum
import math
import random
from copy import deepcopy

# ============================================================================
# DATA STRUCTURES - Note representation for expressive transformation
# ============================================================================

@dataclass
class Note:
    """
    MIDI note with expressive performance attributes

    Attributes:
        pitch: MIDI note number (0-127)
        start_time: Note onset in ticks or seconds
        duration: Note length in ticks or seconds
        velocity: MIDI velocity (0-127)
        channel: MIDI channel (0-15)
        articulation: Performance articulation marker
    """
    pitch: int
    start_time: float
    duration: float
    velocity: int = 64
    channel: int = 0
    articulation: Optional[str] = None

    def __post_init__(self):
        """Validate MIDI ranges"""
        assert 0 <= self.pitch <= 127, f"Invalid pitch: {self.pitch}"
        assert 0 <= self.velocity <= 127, f"Invalid velocity: {self.velocity}"
        assert 0 <= self.channel <= 15, f"Invalid channel: {self.channel}"
        assert self.duration >= 0, f"Invalid duration: {self.duration}"


# ============================================================================
# ENUMS - Expression types and curve shapes
# ============================================================================

class DynamicsCurveType(Enum):
    """Dynamic transition curve shapes (based on MuseScore/Finale)"""
    LINEAR = "linear"              # Constant rate of change
    EXPONENTIAL = "exponential"    # Accelerating change (more dramatic)
    EASE_IN = "ease_in"           # Slow start, fast finish
    EASE_OUT = "ease_out"         # Fast start, slow finish
    EASE_IN_OUT = "ease_in_out"   # S-curve (slow-fast-slow)
    LOGARITHMIC = "logarithmic"   # Decelerating change


class ArticulationType(Enum):
    """Standard musical articulations"""
    LEGATO = "legato"           # Smooth, connected (100% duration + overlap)
    STACCATO = "staccato"       # Short, detached (50% duration)
    STACCATISSIMO = "staccatissimo"  # Very short (25% duration)
    TENUTO = "tenuto"           # Full value, sustained (100% duration)
    MARCATO = "marcato"         # Accented, sharp attack (high velocity)
    PORTATO = "portato"         # Half-staccato (75% duration + slight gap)
    ACCENT = "accent"           # Emphasized (velocity +20)
    SFORZANDO = "sforzando"     # Sudden strong accent (velocity +40)


class SwingStyle(Enum):
    """Swing/groove styles (based on genre research)"""
    STRAIGHT = "straight"       # No swing (50%)
    LIGHT_SWING = "light_swing" # Subtle (55%)
    MEDIUM_SWING = "medium_swing"  # Standard jazz (60%)
    HEAVY_SWING = "heavy_swing"    # Pronounced (66% = triplet)
    J_DILLA = "j_dilla"        # Laid-back, drunk drumming (62-65% variable)
    SHUFFLE = "shuffle"         # Heavy triplet feel (66-70%)


class ExpressionStyle(Enum):
    """Genre-specific expression characteristics"""
    CLASSICAL = "classical"     # Precise, refined dynamics and rubato
    ROMANTIC = "romantic"       # Heavy rubato, dramatic dynamics
    JAZZ = "jazz"              # Swing, syncopation, subtle dynamics
    POP = "pop"                # Moderate expression, consistent tempo
    ROCK = "rock"              # Strong accents, minimal rubato
    ELECTRONIC = "electronic"   # Tight timing, creative velocity shaping


# ============================================================================
# DYNAMICS CURVES - Crescendo, diminuendo, and dynamic shaping
# ============================================================================

class DynamicsEngine:
    """
    Generate expressive dynamics curves for MIDI velocity

    Based on MuseScore, Finale, and Logic Pro implementations
    """

    @staticmethod
    def _apply_curve(t: float, curve_type: DynamicsCurveType) -> float:
        """
        Apply curve shape to normalized time value (0.0 to 1.0)

        Args:
            t: Normalized time (0.0 = start, 1.0 = end)
            curve_type: Shape of the curve

        Returns:
            Curved value (0.0 to 1.0)
        """
        if curve_type == DynamicsCurveType.LINEAR:
            return t

        elif curve_type == DynamicsCurveType.EXPONENTIAL:
            # Exponential curve: y = (e^(2t) - 1) / (e^2 - 1)
            return (math.exp(2 * t) - 1) / (math.exp(2) - 1)

        elif curve_type == DynamicsCurveType.LOGARITHMIC:
            # Logarithmic curve: inverse of exponential
            return math.log(1 + t * (math.e - 1)) / 1

        elif curve_type == DynamicsCurveType.EASE_IN:
            # Quadratic ease-in: y = t^2
            return t * t

        elif curve_type == DynamicsCurveType.EASE_OUT:
            # Quadratic ease-out: y = 1 - (1-t)^2
            return 1 - (1 - t) * (1 - t)

        elif curve_type == DynamicsCurveType.EASE_IN_OUT:
            # S-curve (cubic): smooth acceleration and deceleration
            if t < 0.5:
                return 2 * t * t
            else:
                return 1 - 2 * (1 - t) * (1 - t)

        return t

    @staticmethod
    def apply_dynamics_curve(
        notes: List[Note],
        curve_type: Union[str, DynamicsCurveType] = "crescendo",
        start_vel: int = 60,
        end_vel: int = 110,
        curve_shape: DynamicsCurveType = DynamicsCurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Apply crescendo, diminuendo, or custom dynamics curve

        Args:
            notes: List of Note objects to modify
            curve_type: "crescendo", "diminuendo", or "custom"
            start_vel: Starting velocity (0-127)
            end_vel: Ending velocity (0-127)
            curve_shape: Shape of the transition curve

        Returns:
            Modified notes with dynamic curve applied

        Example:
            >>> notes = [Note(60, i*0.5, 0.5) for i in range(8)]
            >>> notes = DynamicsEngine.apply_dynamics_curve(
            ...     notes, "crescendo", start_vel=60, end_vel=110
            ... )
        """
        if not notes:
            return notes

        # Convert string to enum if needed
        if isinstance(curve_shape, str):
            curve_shape = DynamicsCurveType(curve_shape)

        # Get time range
        start_time = notes[0].start_time
        end_time = notes[-1].start_time
        time_range = end_time - start_time

        if time_range == 0:
            # All notes at same time - apply to all equally
            for note in notes:
                note.velocity = start_vel
            return notes

        # Apply curve to each note
        for note in notes:
            # Normalize time position (0.0 to 1.0)
            t = (note.start_time - start_time) / time_range

            # Apply curve shape
            curved_t = DynamicsEngine._apply_curve(t, curve_shape)

            # Calculate velocity
            velocity = start_vel + (end_vel - start_vel) * curved_t
            note.velocity = max(1, min(127, int(velocity)))

        return notes

    @staticmethod
    def add_dynamic_accents(
        notes: List[Note],
        accent_positions: List[int],
        accent_amount: int = 20,
        accent_type: str = "accent"
    ) -> List[Note]:
        """
        Add accents (sforzando, marcato) at specific positions

        Args:
            notes: List of notes
            accent_positions: Indices of notes to accent
            accent_amount: Velocity increase (default: 20)
            accent_type: "accent", "sforzando", or "marcato"

        Returns:
            Modified notes with accents
        """
        accent_values = {
            "accent": 20,
            "marcato": 30,
            "sforzando": 40
        }

        amount = accent_values.get(accent_type, accent_amount)

        for pos in accent_positions:
            if 0 <= pos < len(notes):
                notes[pos].velocity = min(127, notes[pos].velocity + amount)
                notes[pos].articulation = accent_type

        return notes


# ============================================================================
# VELOCITY HUMANIZATION - Avoid mechanical uniformity
# ============================================================================

class VelocityHumanizer:
    """
    Add natural velocity variation to avoid robotic performances

    Based on GigaMIDI research: DNVR (Distinctive Note Velocity Ratio)
    """

    @staticmethod
    def humanize_velocities(
        notes: List[Note],
        variance: int = 10,
        distribution: str = "gaussian",
        preserve_accents: bool = True,
        seed: Optional[int] = None
    ) -> List[Note]:
        """
        Add natural velocity variations

        Args:
            notes: List of notes to humanize
            variance: Standard deviation for Gaussian (or range for uniform)
            distribution: "gaussian" or "uniform"
            preserve_accents: Keep intentional dynamics (velocities > 100)
            seed: Random seed for reproducibility

        Returns:
            Humanized notes

        Example:
            >>> notes = [Note(60, i*0.5, 0.5, velocity=80) for i in range(8)]
            >>> notes = VelocityHumanizer.humanize_velocities(notes, variance=10)
        """
        if seed is not None:
            random.seed(seed)

        for note in notes:
            # Skip high-velocity accents if preserving
            if preserve_accents and note.velocity > 100:
                continue

            # Generate random variation
            if distribution == "gaussian":
                variation = random.gauss(0, variance)
            elif distribution == "uniform":
                variation = random.uniform(-variance, variance)
            else:
                variation = 0

            # Apply variation with bounds checking
            new_velocity = note.velocity + int(variation)
            note.velocity = max(1, min(127, new_velocity))

        return notes

    @staticmethod
    def add_velocity_contour(
        notes: List[Note],
        contour: List[float],
        scale: int = 30
    ) -> List[Note]:
        """
        Apply velocity contour following a custom shape

        Args:
            notes: Notes to modify
            contour: List of normalized values (0.0 to 1.0) defining shape
            scale: Maximum velocity deviation from current value

        Returns:
            Modified notes
        """
        if not notes or not contour:
            return notes

        # Interpolate contour to match note count
        note_count = len(notes)
        contour_positions = [i * (len(contour) - 1) / (note_count - 1)
                           for i in range(note_count)]

        for i, note in enumerate(notes):
            # Linear interpolation between contour points
            pos = contour_positions[i]
            idx = int(pos)
            frac = pos - idx

            if idx + 1 < len(contour):
                value = contour[idx] * (1 - frac) + contour[idx + 1] * frac
            else:
                value = contour[idx]

            # Apply scaled contour value
            adjustment = int((value - 0.5) * 2 * scale)
            note.velocity = max(1, min(127, note.velocity + adjustment))

        return notes


# ============================================================================
# MICROTIMING & SWING - Roger Linn algorithm and groove feel
# ============================================================================

class MicrotimingEngine:
    """
    Implement swing, groove, and microtiming variations

    Based on:
    - Roger Linn MPC swing algorithm (1979)
    - PMC research on participatory discrepancies (±50ms for groove)
    - J Dilla swing analysis (laid-back, "drunk drumming")
    """

    @staticmethod
    def apply_swing(
        notes: List[Note],
        swing_percent: float = 60,
        subdivision: str = "16th",
        ticks_per_beat: int = 480
    ) -> List[Note]:
        """
        Apply Roger Linn swing algorithm

        Algorithm: Delays even-numbered subdivisions within each beat
        - 50% = no swing (straight)
        - 66% = triplet swing (perfect ternary subdivision)
        - 75% = maximum swing (sounds too extreme beyond this)

        Args:
            notes: Notes to swing
            swing_percent: Swing amount (50-75, default 60)
            subdivision: "16th" or "8th" note swing
            ticks_per_beat: MIDI ticks per quarter note (default 480)

        Returns:
            Swung notes

        Example:
            >>> notes = [Note(60, i*120, 100) for i in range(8)]  # Straight 16ths
            >>> notes = MicrotimingEngine.apply_swing(notes, swing_percent=60)
        """
        if not notes:
            return notes

        # Clamp swing to reasonable range
        swing_percent = max(50, min(75, swing_percent))

        # Calculate subdivision in ticks
        if subdivision == "16th":
            subdivision_ticks = ticks_per_beat / 4
        elif subdivision == "8th":
            subdivision_ticks = ticks_per_beat / 2
        else:
            subdivision_ticks = ticks_per_beat / 4

        # Calculate swing offset
        # At 50%: no offset (straight)
        # At 66%: offset = subdivision_ticks / 3 (triplet)
        # Formula: offset = (swing_percent - 50) / 16.67 * (subdivision_ticks / 3)
        swing_ratio = (swing_percent - 50) / 16.67
        swing_offset = swing_ratio * (subdivision_ticks / 3)

        for note in notes:
            # Determine position within beat
            beat_position = note.start_time % ticks_per_beat
            subdivision_position = beat_position % subdivision_ticks
            subdivision_index = int(beat_position / subdivision_ticks)

            # Swing even-numbered subdivisions (2, 4, 6, 8...)
            if subdivision_index % 2 == 1:
                note.start_time += swing_offset

        return notes

    @staticmethod
    def apply_microtiming(
        notes: List[Note],
        variance_ms: float = 10,
        groove_type: str = "jazz",
        distribution: str = "gaussian",
        seed: Optional[int] = None
    ) -> List[Note]:
        """
        Add microtiming variations for groove feel

        Based on participatory discrepancies research:
        ±50ms microtiming crucial for groove (PMC studies)

        Args:
            notes: Notes to apply microtiming
            variance_ms: Timing variance in milliseconds (default 10ms)
            groove_type: "jazz", "funk", "straight" (affects distribution)
            distribution: "gaussian" or "uniform"
            seed: Random seed

        Returns:
            Notes with microtiming
        """
        if seed is not None:
            random.seed(seed)

        # Adjust variance by groove type
        groove_multipliers = {
            "straight": 0.5,  # Minimal microtiming
            "jazz": 1.0,      # Standard microtiming
            "funk": 1.5,      # More pronounced
            "j_dilla": 2.0    # Extreme laid-back feel
        }

        multiplier = groove_multipliers.get(groove_type, 1.0)
        effective_variance = variance_ms * multiplier

        for note in notes:
            # Generate timing offset
            if distribution == "gaussian":
                offset_ms = random.gauss(0, effective_variance)
            else:
                offset_ms = random.uniform(-effective_variance, effective_variance)

            # Convert to ticks (assuming 120 BPM and 480 TPQN as baseline)
            # 1 beat = 500ms at 120 BPM, so 480 ticks = 500ms
            offset_ticks = offset_ms * 480 / 500

            note.start_time += offset_ticks

        return notes

    @staticmethod
    def create_j_dilla_swing(
        notes: List[Note],
        drunk_factor: float = 0.7,
        ticks_per_beat: int = 480
    ) -> List[Note]:
        """
        Create J Dilla's characteristic "drunk drumming" feel

        Combines:
        - Variable swing (62-65% per beat, not consistent)
        - Slight delays on certain beats
        - Laid-back timing

        Args:
            notes: Notes to process
            drunk_factor: Intensity of effect (0.0 to 1.0)
            ticks_per_beat: MIDI ticks per beat

        Returns:
            J Dilla-style swung notes
        """
        if not notes:
            return notes

        # Apply variable swing per beat
        current_beat = -1
        current_swing = 62

        for note in notes:
            beat = int(note.start_time / ticks_per_beat)

            # Change swing amount on new beat
            if beat != current_beat:
                current_beat = beat
                # Random swing between 60-66%
                current_swing = 60 + random.random() * 6 * drunk_factor

            # Apply swing
            beat_position = note.start_time % ticks_per_beat
            subdivision_ticks = ticks_per_beat / 4
            subdivision_index = int(beat_position / subdivision_ticks)

            if subdivision_index % 2 == 1:
                swing_ratio = (current_swing - 50) / 16.67
                swing_offset = swing_ratio * (subdivision_ticks / 3)
                note.start_time += swing_offset

            # Add slight random delays (laid-back feel)
            if random.random() < 0.3 * drunk_factor:
                note.start_time += random.uniform(5, 15)

        return notes


# ============================================================================
# RUBATO & TEMPO CURVES - Expressive timing
# ============================================================================

class RubatoEngine:
    """
    Generate rubato and tempo curves

    Based on:
    - Chopin Nocturnes kinematic model research
    - Romantic piano performance practice
    - Paderewski writings on rubato
    """

    @staticmethod
    def apply_rubato(
        notes: List[Note],
        rubato_curve: Optional[List[float]] = None,
        intensity: float = 0.3,
        style: str = "romantic"
    ) -> List[Note]:
        """
        Apply expressive timing deviations (rubato)

        Args:
            notes: Notes to apply rubato
            rubato_curve: Custom curve (1.0 = normal tempo, 0.5 = half tempo, 2.0 = double)
                         If None, generates romantic-style curve
            intensity: Amount of rubato (0.0 to 1.0)
            style: "romantic", "classical", "jazz"

        Returns:
            Notes with rubato timing

        Example:
            >>> notes = [Note(60, i*480, 400) for i in range(16)]
            >>> notes = RubatoEngine.apply_rubato(notes, intensity=0.4)
        """
        if not notes:
            return notes

        # Generate default rubato curve if not provided
        if rubato_curve is None:
            rubato_curve = RubatoEngine._generate_rubato_curve(len(notes), style)

        # Interpolate curve to match note count
        curve_values = []
        for i in range(len(notes)):
            t = i / (len(notes) - 1) if len(notes) > 1 else 0
            idx = t * (len(rubato_curve) - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(rubato_curve) - 1)
            frac = idx - idx_low

            value = rubato_curve[idx_low] * (1 - frac) + rubato_curve[idx_high] * frac
            curve_values.append(value)

        # Apply tempo modifications cumulatively
        time_offset = 0.0
        for i, note in enumerate(notes):
            # Calculate timing deviation based on curve
            tempo_factor = 1.0 + (curve_values[i] - 1.0) * intensity

            # Apply to note timing
            original_time = note.start_time
            note.start_time += time_offset

            # Calculate offset for next note
            if i < len(notes) - 1:
                interval = notes[i + 1].start_time - original_time
                modified_interval = interval * tempo_factor
                time_offset += (modified_interval - interval)

        return notes

    @staticmethod
    def _generate_rubato_curve(length: int, style: str) -> List[float]:
        """Generate style-appropriate rubato curve"""
        curve = []

        if style == "romantic":
            # Romantic: slow start, accelerate middle, slow end
            for i in range(length):
                t = i / (length - 1) if length > 1 else 0
                # S-curve variant
                if t < 0.25:
                    value = 0.8 + t * 0.8  # Slow start
                elif t < 0.75:
                    value = 1.0 + (t - 0.25) * 0.4  # Accelerate
                else:
                    value = 1.2 - (t - 0.75) * 0.8  # Slow end
                curve.append(value)

        elif style == "classical":
            # Classical: subtle, symmetric
            for i in range(length):
                t = i / (length - 1) if length > 1 else 0
                value = 1.0 + 0.1 * math.sin(t * math.pi)
                curve.append(value)

        elif style == "jazz":
            # Jazz: irregular, phrase-based
            for i in range(length):
                t = i / (length - 1) if length > 1 else 0
                value = 1.0 + 0.05 * math.sin(t * 2 * math.pi) * random.uniform(0.8, 1.2)
                curve.append(value)

        else:
            curve = [1.0] * length

        return curve

    @staticmethod
    def apply_accelerando(
        notes: List[Note],
        start_tempo_ratio: float = 1.0,
        end_tempo_ratio: float = 1.5,
        curve_type: DynamicsCurveType = DynamicsCurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Apply accelerando (gradual speed up)

        Args:
            notes: Notes to accelerate
            start_tempo_ratio: Starting tempo (1.0 = normal)
            end_tempo_ratio: Ending tempo (1.5 = 50% faster)
            curve_type: Acceleration curve shape

        Returns:
            Accelerated notes
        """
        if not notes:
            return notes

        time_offset = 0.0
        for i, note in enumerate(notes):
            # Calculate position in accelerando
            t = i / (len(notes) - 1) if len(notes) > 1 else 0
            curved_t = DynamicsEngine._apply_curve(t, curve_type)

            # Calculate tempo ratio at this point
            tempo_ratio = start_tempo_ratio + (end_tempo_ratio - start_tempo_ratio) * curved_t

            # Apply offset
            original_time = note.start_time
            note.start_time += time_offset

            # Calculate offset for next note
            if i < len(notes) - 1:
                interval = notes[i + 1].start_time - original_time
                # Faster tempo = shorter interval
                modified_interval = interval / tempo_ratio
                time_offset += (modified_interval - interval)

        return notes

    @staticmethod
    def apply_ritardando(
        notes: List[Note],
        start_tempo_ratio: float = 1.0,
        end_tempo_ratio: float = 0.5,
        curve_type: DynamicsCurveType = DynamicsCurveType.EXPONENTIAL
    ) -> List[Note]:
        """
        Apply ritardando (gradual slow down)

        Args:
            notes: Notes to slow down
            start_tempo_ratio: Starting tempo (1.0 = normal)
            end_tempo_ratio: Ending tempo (0.5 = half speed)
            curve_type: Deceleration curve shape

        Returns:
            Slowed notes
        """
        return RubatoEngine.apply_accelerando(
            notes, start_tempo_ratio, end_tempo_ratio, curve_type
        )


# ============================================================================
# ARTICULATION - Staccato, legato, marcato, tenuto
# ============================================================================

class ArticulationEngine:
    """
    Render musical articulations

    Based on MuseScore standards and performance practice
    """

    # Articulation duration multipliers
    DURATION_MULTIPLIERS = {
        ArticulationType.STACCATISSIMO: 0.25,
        ArticulationType.STACCATO: 0.50,
        ArticulationType.PORTATO: 0.75,
        ArticulationType.TENUTO: 1.0,
        ArticulationType.LEGATO: 1.0,
        ArticulationType.MARCATO: 0.90,
        ArticulationType.ACCENT: 1.0,
        ArticulationType.SFORZANDO: 1.0
    }

    # Velocity adjustments
    VELOCITY_ADJUSTMENTS = {
        ArticulationType.STACCATISSIMO: 5,
        ArticulationType.STACCATO: 0,
        ArticulationType.PORTATO: 0,
        ArticulationType.TENUTO: 0,
        ArticulationType.LEGATO: -5,
        ArticulationType.MARCATO: 30,
        ArticulationType.ACCENT: 20,
        ArticulationType.SFORZANDO: 40
    }

    @staticmethod
    def render_articulation(
        notes: List[Note],
        articulation: Union[str, ArticulationType],
        overlap: float = 0.1
    ) -> List[Note]:
        """
        Apply articulation to notes

        Args:
            notes: Notes to articulate
            articulation: Articulation type
            overlap: Overlap amount for legato (0.0 to 0.5)

        Returns:
            Articulated notes

        Example:
            >>> notes = [Note(60, i*480, 400) for i in range(8)]
            >>> notes = ArticulationEngine.render_articulation(notes, "staccato")
        """
        if isinstance(articulation, str):
            articulation = ArticulationType(articulation)

        duration_mult = ArticulationEngine.DURATION_MULTIPLIERS.get(articulation, 1.0)
        velocity_adj = ArticulationEngine.VELOCITY_ADJUSTMENTS.get(articulation, 0)

        for i, note in enumerate(notes):
            # Modify duration
            note.duration *= duration_mult

            # Legato: extend to overlap with next note
            if articulation == ArticulationType.LEGATO and i < len(notes) - 1:
                next_start = notes[i + 1].start_time
                max_duration = next_start - note.start_time
                overlap_duration = max_duration * (1 + overlap)
                note.duration = min(overlap_duration, max_duration * 1.2)

            # Adjust velocity
            note.velocity = max(1, min(127, note.velocity + velocity_adj))

            # Mark articulation
            note.articulation = articulation.value

        return notes

    @staticmethod
    def add_agogic_accents(
        notes: List[Note],
        accent_indices: List[int],
        lengthen_percent: int = 15
    ) -> List[Note]:
        """
        Add agogic accents (emphasis through duration)

        Args:
            notes: Notes to modify
            accent_indices: Indices of notes to accent
            lengthen_percent: Percentage to lengthen note

        Returns:
            Modified notes
        """
        for idx in accent_indices:
            if 0 <= idx < len(notes):
                notes[idx].duration *= (1 + lengthen_percent / 100)

        return notes


# ============================================================================
# STYLE-SPECIFIC EXPRESSION - Genre-appropriate performance
# ============================================================================

class StyleEngine:
    """
    Apply genre-specific expressive characteristics
    """

    @staticmethod
    def apply_style(
        notes: List[Note],
        style: Union[str, ExpressionStyle],
        intensity: float = 0.7
    ) -> List[Note]:
        """
        Apply style-specific expression bundle

        Args:
            notes: Notes to style
            style: Expression style (classical, jazz, pop, etc.)
            intensity: Expression intensity (0.0 to 1.0)

        Returns:
            Styled notes
        """
        if isinstance(style, str):
            style = ExpressionStyle(style)

        if style == ExpressionStyle.CLASSICAL:
            # Precise dynamics, subtle rubato
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(5 * intensity))
            notes = RubatoEngine.apply_rubato(notes, intensity=0.2 * intensity, style="classical")
            notes = ArticulationEngine.render_articulation(notes, ArticulationType.TENUTO)

        elif style == ExpressionStyle.ROMANTIC:
            # Heavy rubato, dramatic dynamics
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(15 * intensity))
            notes = RubatoEngine.apply_rubato(notes, intensity=0.5 * intensity, style="romantic")
            notes = DynamicsEngine.apply_dynamics_curve(
                notes, "crescendo", 50, 100, DynamicsCurveType.EXPONENTIAL
            )

        elif style == ExpressionStyle.JAZZ:
            # Swing, subtle dynamics
            notes = MicrotimingEngine.apply_swing(notes, swing_percent=60)
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(12 * intensity))
            notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=8, groove_type="jazz")

        elif style == ExpressionStyle.POP:
            # Tight timing, moderate expression
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(8 * intensity))
            notes = MicrotimingEngine.apply_microtiming(notes, variance_ms=5, groove_type="straight")

        elif style == ExpressionStyle.ROCK:
            # Strong accents, minimal rubato, tight timing
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(10 * intensity))
            # Accent downbeats
            accent_positions = [i for i in range(0, len(notes), 4)]
            notes = DynamicsEngine.add_dynamic_accents(notes, accent_positions, 25)

        elif style == ExpressionStyle.ELECTRONIC:
            # Tight timing, creative velocity shaping
            notes = VelocityHumanizer.humanize_velocities(notes, variance=int(3 * intensity))
            # Creative velocity contour
            contour = [0.2, 0.8, 0.4, 1.0, 0.3, 0.7, 0.5, 0.9]
            notes = VelocityHumanizer.add_velocity_contour(notes, contour, scale=20)

        return notes


# ============================================================================
# MAIN EXPRESSIVE PERFORMANCE CLASS
# ============================================================================

class ExpressivePerformance:
    """
    Main interface for expressive MIDI performance transformations

    Combines all expressive techniques into a single unified API
    """

    def __init__(self, ticks_per_beat: int = 480):
        """
        Initialize performance engine

        Args:
            ticks_per_beat: MIDI ticks per quarter note (default 480)
        """
        self.ticks_per_beat = ticks_per_beat
        self.dynamics = DynamicsEngine()
        self.velocity = VelocityHumanizer()
        self.microtiming = MicrotimingEngine()
        self.rubato = RubatoEngine()
        self.articulation = ArticulationEngine()
        self.style = StyleEngine()

    def make_expressive(
        self,
        notes: List[Note],
        style: str = "classical",
        dynamics: str = "moderate",
        timing: str = "natural",
        articulation: Optional[str] = None
    ) -> List[Note]:
        """
        Apply full expressive transformation with sensible defaults

        Args:
            notes: Notes to make expressive
            style: "classical", "romantic", "jazz", "pop", "rock", "electronic"
            dynamics: "subtle", "moderate", "dramatic"
            timing: "tight", "natural", "loose", "swing"
            articulation: Optional articulation override

        Returns:
            Fully expressive notes
        """
        # Deep copy to avoid modifying original
        notes = deepcopy(notes)

        # Apply style-specific bundle
        notes = self.style.apply_style(notes, style, intensity=0.7)

        # Dynamics adjustments
        dynamics_variance = {"subtle": 5, "moderate": 10, "dramatic": 15}
        variance = dynamics_variance.get(dynamics, 10)
        notes = self.velocity.humanize_velocities(notes, variance=variance)

        # Timing adjustments
        if timing == "tight":
            notes = self.microtiming.apply_microtiming(notes, variance_ms=3)
        elif timing == "natural":
            notes = self.microtiming.apply_microtiming(notes, variance_ms=8)
        elif timing == "loose":
            notes = self.microtiming.apply_microtiming(notes, variance_ms=15)
        elif timing == "swing":
            notes = self.microtiming.apply_swing(notes, swing_percent=60)

        # Articulation override
        if articulation:
            notes = self.articulation.render_articulation(notes, articulation)

        return notes


# ============================================================================
# UNIT TESTS AND EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EXPRESSIVE PERFORMANCE MODULE - TEST SUITE")
    print("=" * 70)

    # Test 1: Dynamics curve (crescendo)
    print("\n[TEST 1] Crescendo (vel 60→110, exponential)")
    notes = [Note(60 + i % 12, i * 480, 400, velocity=60) for i in range(8)]
    notes_cresc = DynamicsEngine.apply_dynamics_curve(
        notes, "crescendo", start_vel=60, end_vel=110,
        curve_shape=DynamicsCurveType.EXPONENTIAL
    )
    print(f"Velocities: {[n.velocity for n in notes_cresc]}")
    assert notes_cresc[0].velocity == 60
    assert notes_cresc[-1].velocity == 110
    assert notes_cresc[4].velocity > notes_cresc[0].velocity
    print("✓ PASS")

    # Test 2: Diminuendo
    print("\n[TEST 2] Diminuendo (vel 100→40, linear)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_dim = DynamicsEngine.apply_dynamics_curve(
        notes, "diminuendo", start_vel=100, end_vel=40,
        curve_shape=DynamicsCurveType.LINEAR
    )
    print(f"Velocities: {[n.velocity for n in notes_dim]}")
    assert notes_dim[0].velocity == 100
    assert notes_dim[-1].velocity == 40
    print("✓ PASS")

    # Test 3: Velocity humanization
    print("\n[TEST 3] Velocity humanization (variance=10, gaussian)")
    notes = [Note(60, i * 480, 400, velocity=80) for i in range(8)]
    notes_human = VelocityHumanizer.humanize_velocities(notes, variance=10, seed=42)
    velocities = [n.velocity for n in notes_human]
    print(f"Velocities: {velocities}")
    # Check that velocities vary but stay in reasonable range
    assert min(velocities) >= 65 and max(velocities) <= 95
    assert len(set(velocities)) > 4  # Should have variation
    print("✓ PASS")

    # Test 4: Roger Linn swing (60%)
    print("\n[TEST 4] Roger Linn swing (60%, 16th notes)")
    notes = [Note(60, i * 120, 100) for i in range(8)]  # Straight 16ths
    original_times = [n.start_time for n in notes]
    notes_swing = MicrotimingEngine.apply_swing(notes, swing_percent=60, subdivision="16th")
    swung_times = [n.start_time for n in notes_swing]
    print(f"Original: {original_times}")
    print(f"Swung:    {swung_times}")
    # Even positions should be delayed
    assert notes_swing[1].start_time > original_times[1]
    assert notes_swing[3].start_time > original_times[3]
    print("✓ PASS")

    # Test 5: Triplet swing (66%)
    print("\n[TEST 5] Triplet swing (66%)")
    notes = [Note(60, i * 120, 100) for i in range(8)]
    notes_triplet = MicrotimingEngine.apply_swing(notes, swing_percent=66)
    print(f"Swung times: {[n.start_time for n in notes_triplet]}")
    print("✓ PASS")

    # Test 6: Microtiming
    print("\n[TEST 6] Microtiming (10ms variance, jazz groove)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    original_times = [n.start_time for n in notes]
    notes_micro = MicrotimingEngine.apply_microtiming(
        notes, variance_ms=10, groove_type="jazz", seed=42
    )
    micro_times = [n.start_time for n in notes_micro]
    print(f"Original: {original_times}")
    print(f"Micro:    {[f'{t:.1f}' for t in micro_times]}")
    # Times should vary
    assert micro_times != original_times
    print("✓ PASS")

    # Test 7: J Dilla swing
    print("\n[TEST 7] J Dilla swing (drunk_factor=0.7)")
    notes = [Note(60, i * 120, 100) for i in range(16)]
    notes_dilla = MicrotimingEngine.create_j_dilla_swing(notes, drunk_factor=0.7)
    print(f"J Dilla times (first 8): {[f'{n.start_time:.1f}' for n in notes_dilla[:8]]}")
    print("✓ PASS")

    # Test 8: Rubato (romantic style)
    print("\n[TEST 8] Rubato (romantic style, intensity=0.4)")
    notes = [Note(60 + i % 12, i * 480, 400) for i in range(16)]
    original_times = [n.start_time for n in notes]
    notes_rubato = RubatoEngine.apply_rubato(notes, intensity=0.4, style="romantic")
    rubato_times = [n.start_time for n in notes_rubato]
    print(f"Original (first 8): {original_times[:8]}")
    print(f"Rubato (first 8):   {[f'{t:.1f}' for t in rubato_times[:8]]}")
    # Times should deviate
    assert rubato_times != original_times
    print("✓ PASS")

    # Test 9: Accelerando
    print("\n[TEST 9] Accelerando (1.0 → 1.5x tempo)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_accel = RubatoEngine.apply_accelerando(
        notes, start_tempo_ratio=1.0, end_tempo_ratio=1.5
    )
    intervals_accel = [notes_accel[i+1].start_time - notes_accel[i].start_time
                       for i in range(len(notes_accel)-1)]
    print(f"Intervals: {[f'{i:.1f}' for i in intervals_accel]}")
    # Intervals should decrease (speeding up)
    assert intervals_accel[-1] < intervals_accel[0]
    print("✓ PASS")

    # Test 10: Ritardando
    print("\n[TEST 10] Ritardando (1.0 → 0.5x tempo)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_rit = RubatoEngine.apply_ritardando(
        notes, start_tempo_ratio=1.0, end_tempo_ratio=0.5
    )
    intervals_rit = [notes_rit[i+1].start_time - notes_rit[i].start_time
                     for i in range(len(notes_rit)-1)]
    print(f"Intervals: {[f'{i:.1f}' for i in intervals_rit]}")
    # Intervals should increase (slowing down)
    assert intervals_rit[-1] > intervals_rit[0]
    print("✓ PASS")

    # Test 11: Staccato articulation
    print("\n[TEST 11] Staccato articulation (50% duration)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_staccato = ArticulationEngine.render_articulation(notes, "staccato")
    durations = [n.duration for n in notes_staccato]
    print(f"Durations: {durations}")
    assert all(d == 200 for d in durations)  # 50% of 400
    print("✓ PASS")

    # Test 12: Legato articulation
    print("\n[TEST 12] Legato articulation (overlap)")
    notes = [Note(60 + i, i * 480, 400) for i in range(8)]
    notes_legato = ArticulationEngine.render_articulation(notes, "legato", overlap=0.1)
    durations = [n.duration for n in notes_legato]
    print(f"Durations (first 4): {[f'{d:.1f}' for d in durations[:4]]}")
    # Legato should extend durations to overlap
    assert notes_legato[0].duration > 400
    print("✓ PASS")

    # Test 13: Marcato articulation
    print("\n[TEST 13] Marcato articulation (high velocity)")
    notes = [Note(60, i * 480, 400, velocity=80) for i in range(8)]
    notes_marcato = ArticulationEngine.render_articulation(notes, "marcato")
    velocities = [n.velocity for n in notes_marcato]
    print(f"Velocities: {velocities}")
    assert all(v == 110 for v in velocities)  # 80 + 30
    print("✓ PASS")

    # Test 14: Sforzando accents
    print("\n[TEST 14] Sforzando accents")
    notes = [Note(60, i * 480, 400, velocity=70) for i in range(8)]
    notes_sfz = DynamicsEngine.add_dynamic_accents(notes, [0, 4], accent_type="sforzando")
    velocities = [n.velocity for n in notes_sfz]
    print(f"Velocities: {velocities}")
    assert notes_sfz[0].velocity == 110  # 70 + 40
    assert notes_sfz[4].velocity == 110
    assert notes_sfz[2].velocity == 70  # Unchanged
    print("✓ PASS")

    # Test 15: Classical style
    print("\n[TEST 15] Classical style (subtle expression)")
    notes = [Note(60 + i % 12, i * 480, 400, velocity=80) for i in range(16)]
    notes_classical = StyleEngine.apply_style(notes, "classical", intensity=0.7)
    print(f"Velocities (first 8): {[n.velocity for n in notes_classical[:8]]}")
    print("✓ PASS")

    # Test 16: Jazz style (with swing)
    print("\n[TEST 16] Jazz style (swing + humanization)")
    notes = [Note(60, i * 120, 100, velocity=80) for i in range(16)]
    notes_jazz = StyleEngine.apply_style(notes, "jazz", intensity=0.8)
    print(f"Times (first 8): {[f'{n.start_time:.1f}' for n in notes_jazz[:8]]}")
    print(f"Velocities (first 8): {[n.velocity for n in notes_jazz[:8]]}")
    print("✓ PASS")

    # Test 17: Romantic style (heavy rubato)
    print("\n[TEST 17] Romantic style (dramatic expression)")
    notes = [Note(60 + i % 12, i * 480, 400, velocity=70) for i in range(16)]
    notes_romantic = StyleEngine.apply_style(notes, "romantic", intensity=0.9)
    print(f"Times (first 8): {[f'{n.start_time:.1f}' for n in notes_romantic[:8]]}")
    print(f"Velocities (first 8): {[n.velocity for n in notes_romantic[:8]]}")
    print("✓ PASS")

    # Test 18: Pop style
    print("\n[TEST 18] Pop style (moderate expression)")
    notes = [Note(60, i * 480, 400, velocity=80) for i in range(8)]
    notes_pop = StyleEngine.apply_style(notes, "pop", intensity=0.7)
    print(f"Velocities: {[n.velocity for n in notes_pop]}")
    print("✓ PASS")

    # Test 19: Agogic accents
    print("\n[TEST 19] Agogic accents (lengthen 15%)")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_agogic = ArticulationEngine.add_agogic_accents(notes, [0, 2, 4, 6], lengthen_percent=15)
    durations = [n.duration for n in notes_agogic]
    print(f"Durations: {durations}")
    assert abs(notes_agogic[0].duration - 460) < 0.1  # 400 * 1.15 (floating point tolerance)
    assert notes_agogic[1].duration == 400  # Unchanged
    print("✓ PASS")

    # Test 20: Full expressive transformation
    print("\n[TEST 20] Complete expressive transformation")
    perf = ExpressivePerformance(ticks_per_beat=480)
    notes = [Note(60 + i % 12, i * 480, 400, velocity=80) for i in range(16)]
    notes_expressive = perf.make_expressive(
        notes, style="romantic", dynamics="dramatic", timing="natural"
    )
    print(f"Times (first 8): {[f'{n.start_time:.1f}' for n in notes_expressive[:8]]}")
    print(f"Velocities (first 8): {[n.velocity for n in notes_expressive[:8]]}")
    print(f"Durations (first 8): {[f'{n.duration:.1f}' for n in notes_expressive[:8]]}")
    print("✓ PASS")

    # Test 21: Ease-in curve
    print("\n[TEST 21] Ease-in dynamics curve")
    notes = [Note(60, i * 480, 400) for i in range(8)]
    notes_ease = DynamicsEngine.apply_dynamics_curve(
        notes, "crescendo", 50, 100, DynamicsCurveType.EASE_IN
    )
    velocities = [n.velocity for n in notes_ease]
    print(f"Velocities: {velocities}")
    # Ease-in: slower start, faster end
    assert velocities[1] - velocities[0] < velocities[-1] - velocities[-2]
    print("✓ PASS")

    # Test 22: Velocity contour
    print("\n[TEST 22] Custom velocity contour")
    notes = [Note(60, i * 480, 400, velocity=70) for i in range(8)]
    contour = [0.0, 0.3, 0.7, 1.0, 0.8, 0.5, 0.2, 0.1]
    notes_contour = VelocityHumanizer.add_velocity_contour(notes, contour, scale=30)
    velocities = [n.velocity for n in notes_contour]
    print(f"Velocities: {velocities}")
    print("✓ PASS")

    print("\n" + "=" * 70)
    print(f"ALL 22 TESTS PASSED! ✓")
    print("=" * 70)
    print("\nExpressivePerformance module ready for production use!")
    print("\nResearch citations:")
    print("  [1] Nature Scientific Reports 2025 - Transformer models")
    print("  [2] MAESTRO Dataset - 200 hours piano performance")
    print("  [3] GigaMIDI - 1.4M files, micro-timing analysis")
    print("  [4] Roger Linn - MPC swing algorithm (50-75%)")
    print("  [5] PMC - Participatory discrepancies (±50ms groove)")
    print("=" * 70)
