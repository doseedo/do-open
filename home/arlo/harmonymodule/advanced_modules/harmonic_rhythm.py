#!/usr/bin/env python3
"""
Harmonic Rhythm & Progression Pacing Module

Control the rate and pacing of chord changes to create musical tension,
release, and forward momentum. This module provides sophisticated tools
for managing harmonic density and temporal organization of chord progressions.

Features:
- Harmonic rhythm generation (slow, medium, fast)
- Chord density control and analysis
- Tension/release via harmonic pacing
- Genre-appropriate rhythm patterns
- Harmonic acceleration/deceleration
- Suspension and anticipation timing
- Analysis of existing progressions

Based on research from:
- Piston, Walter: "Harmony" (Classical harmonic rhythm principles)
- Levine, Mark: "The Jazz Theory Book" (Bebop harmonic rhythm, 2 chords/bar)
- Caplin, William: "Classical Form" (Phrase structure and cadential acceleration)
- EURASIP Journal 2024: "Generating chord progression from melody with flexible
  harmonic rhythm and controllable harmonic density" (AutoHarmonizer)
- Music Theory pedagogy: harmonic rhythm accelerates approaching cadences

Author: Agent 19 - Harmonic Rhythm Specialist
Date: 2025
"""

from typing import List, Dict, Tuple, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import random
import math
from collections import defaultdict

# Try to import numpy, fall back to Python built-ins if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback implementations
    class np:
        @staticmethod
        def linspace(start, stop, num):
            if num == 1:
                return [start]
            step = (stop - start) / (num - 1)
            return [start + step * i for i in range(num)]

        @staticmethod
        def interp(x, xp, fp):
            """Simple linear interpolation"""
            result = []
            for xi in x:
                # Find surrounding points
                for i in range(len(xp) - 1):
                    if xp[i] <= xi <= xp[i+1]:
                        # Linear interpolation
                        t = (xi - xp[i]) / (xp[i+1] - xp[i])
                        result.append(fp[i] + t * (fp[i+1] - fp[i]))
                        break
                else:
                    # Outside range
                    if xi < xp[0]:
                        result.append(fp[0])
                    else:
                        result.append(fp[-1])
            return result

        @staticmethod
        def var(data):
            """Calculate variance"""
            if not data:
                return 0.0
            mean_val = sum(data) / len(data)
            return sum((x - mean_val) ** 2 for x in data) / len(data)

        @staticmethod
        def sin(x):
            if isinstance(x, list):
                return [math.sin(xi) for xi in x]
            return math.sin(x)

        @staticmethod
        def concatenate(arrays):
            """Concatenate arrays"""
            result = []
            for arr in arrays:
                if isinstance(arr, (list, tuple)):
                    result.extend(arr)
                else:
                    result.append(arr)
            return result

        @staticmethod
        def full(size, value):
            """Create array filled with value"""
            return [value] * size

        pi = math.pi


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class HarmonicDensity(Enum):
    """Harmonic density levels (chords per measure)"""
    VERY_SPARSE = 0.25      # 1 chord per 4 measures
    SPARSE = 0.5            # 1 chord per 2 measures
    LOW = 1.0               # 1 chord per measure
    MEDIUM = 2.0            # 2 chords per measure
    HIGH = 4.0              # 4 chords per measure (every beat in 4/4)
    VERY_HIGH = 8.0         # 8 chords per measure (every 8th note)
    EXTREME = 16.0          # 16 chords per measure (bebop/rapid changes)


class GenreStyle(Enum):
    """Genre-specific harmonic rhythm patterns"""
    POP = "pop"                      # 1-2 bars per chord, simple patterns
    ROCK = "rock"                    # 1-2 bars per chord, power chords sustained
    JAZZ_SWING = "jazz_swing"        # 1-2 chords per bar, moderate density
    BEBOP = "bebop"                  # 2-4 chords per bar, high density
    CLASSICAL_BAROQUE = "baroque"    # Steady harmonic rhythm, ornamental surface
    CLASSICAL_ROMANTIC = "romantic"  # Variable density, dramatic changes
    FUNK = "funk"                    # Sustained chords, 1 per 2-4 bars
    BLUES = "blues"                  # 12-bar form, 1-2 chords per bar
    GOSPEL = "gospel"                # Frequent changes, 2-4 per bar
    ELECTRONIC = "electronic"        # Minimal changes, 1 per 4-8 bars
    MINIMALIST = "minimalist"        # Very sparse, 1 per 8+ bars


class TensionCurveType(Enum):
    """Types of tension curves for harmonic pacing"""
    LINEAR_INCREASE = "linear_increase"           # Steady increase
    LINEAR_DECREASE = "linear_decrease"           # Steady decrease
    EXPONENTIAL_INCREASE = "exponential_increase" # Accelerating increase
    EXPONENTIAL_DECREASE = "exponential_decrease" # Decelerating decrease
    ARCH = "arch"                                 # Rise to peak, then fall
    VALLEY = "valley"                             # Fall to valley, then rise
    WAVE = "wave"                                 # Sinusoidal oscillation
    CADENTIAL = "cadential"                       # Accelerate to cadence


@dataclass
class ChordDuration:
    """Represents duration of a single chord"""
    chord_index: int                # Index in progression
    start_beat: float              # Starting beat position
    duration_beats: float          # Duration in beats
    measure: int                   # Which measure (0-indexed)
    beat_in_measure: float        # Beat position within measure (0-indexed)
    is_suspension: bool = False   # Is this a suspended chord?
    is_anticipation: bool = False # Is this an anticipated chord?
    tension_level: float = 0.5    # Tension level (0.0-1.0)


@dataclass
class HarmonicRhythmPattern:
    """Complete harmonic rhythm pattern"""
    chord_durations: List[ChordDuration]
    total_measures: int
    beats_per_measure: int
    average_density: float         # Average chords per measure
    density_variance: float        # Variance in density
    genre_style: Optional[GenreStyle] = None
    tension_curve: Optional[List[float]] = None


# ============================================================================
# MAIN HARMONIC RHYTHM CLASS
# ============================================================================

class HarmonicRhythm:
    """
    Control harmonic rhythm and chord pacing

    This class provides comprehensive tools for managing the temporal
    organization of chord progressions, including:
    - Generating harmonic rhythms at various densities
    - Analyzing existing progressions
    - Creating tension/release through pacing
    - Genre-appropriate patterns
    - Dynamic acceleration/deceleration

    Examples:
        >>> hr = HarmonicRhythm()
        >>> # Generate medium density rhythm (2 chords/bar, 8 measures)
        >>> pattern = hr.generate_harmonic_rhythm(
        ...     density="medium",
        ...     total_measures=8
        ... )
        >>>
        >>> # Create genre-specific pattern
        >>> bebop_pattern = hr.create_genre_appropriate_rhythm(
        ...     genre="bebop",
        ...     measures=16
        ... )
        >>>
        >>> # Apply acceleration to cadence
        >>> accel_pattern = hr.apply_harmonic_acceleration(
        ...     progression_length=12,
        ...     start_density=1,
        ...     end_density=4
        ... )
    """

    def __init__(
        self,
        beats_per_measure: int = 4,
        beat_unit: int = 4,  # 4 = quarter note
        default_tempo: float = 120.0
    ):
        """
        Initialize HarmonicRhythm engine

        Args:
            beats_per_measure: Number of beats per measure (default 4)
            beat_unit: Beat unit (4 = quarter, 8 = eighth, etc.)
            default_tempo: Default tempo in BPM
        """
        self.beats_per_measure = beats_per_measure
        self.beat_unit = beat_unit
        self.default_tempo = default_tempo

        # Genre-specific density mappings (average chords per measure)
        self.genre_densities = {
            GenreStyle.POP: (0.5, 1.0),               # 0.5-1 chord per measure
            GenreStyle.ROCK: (0.5, 1.0),              # 0.5-1 chord per measure
            GenreStyle.JAZZ_SWING: (1.0, 2.0),        # 1-2 chords per measure
            GenreStyle.BEBOP: (2.0, 4.0),             # 2-4 chords per measure
            GenreStyle.CLASSICAL_BAROQUE: (1.0, 2.0), # 1-2 chords per measure
            GenreStyle.CLASSICAL_ROMANTIC: (0.5, 2.0),# Variable 0.5-2
            GenreStyle.FUNK: (0.25, 0.5),             # 1 chord per 2-4 bars
            GenreStyle.BLUES: (1.0, 1.5),             # 1-1.5 chords per measure
            GenreStyle.GOSPEL: (2.0, 4.0),            # 2-4 chords per measure
            GenreStyle.ELECTRONIC: (0.125, 0.25),     # 1 chord per 4-8 bars
            GenreStyle.MINIMALIST: (0.0625, 0.125),   # 1 chord per 8-16 bars
        }

    def generate_harmonic_rhythm(
        self,
        density: Union[str, float, HarmonicDensity] = "medium",
        total_measures: int = 8,
        variation: float = 0.2,
        prefer_downbeats: bool = True
    ) -> HarmonicRhythmPattern:
        """
        Generate harmonic rhythm pattern with specified density

        Args:
            density: Density level - "slow"/"low", "medium", "fast"/"high",
                    or float (chords per measure), or HarmonicDensity enum
            total_measures: Total number of measures
            variation: Amount of variation in chord durations (0.0-1.0)
            prefer_downbeats: If True, place more chords on strong beats

        Returns:
            HarmonicRhythmPattern with chord durations
        """
        # Convert density to numeric value
        if isinstance(density, str):
            density_map = {
                "very_sparse": 0.25, "sparse": 0.5,
                "slow": 0.5, "low": 1.0,
                "medium": 2.0, "high": 4.0,
                "fast": 4.0, "very_high": 8.0,
                "extreme": 16.0
            }
            density_value = density_map.get(density.lower(), 2.0)
        elif isinstance(density, HarmonicDensity):
            density_value = density.value
        else:
            density_value = float(density)

        total_beats = total_measures * self.beats_per_measure
        target_num_chords = int(density_value * total_measures)
        target_num_chords = max(1, target_num_chords)  # At least 1 chord

        # Generate chord change positions
        chord_durations = []
        current_beat = 0.0
        chord_index = 0

        while current_beat < total_beats and chord_index < target_num_chords:
            # Calculate base duration
            base_duration = total_beats / target_num_chords

            # Add variation
            if variation > 0:
                vary_amount = base_duration * variation
                duration = base_duration + random.uniform(-vary_amount, vary_amount)
                duration = max(0.5, duration)  # Minimum half beat
            else:
                duration = base_duration

            # Snap to beat if prefer_downbeats
            if prefer_downbeats and chord_index > 0:
                # Round to nearest beat or half-beat
                next_beat = current_beat + duration
                rounded = round(next_beat * 2) / 2  # Round to nearest 0.5
                duration = rounded - current_beat
                duration = max(0.5, duration)

            # Don't exceed total beats
            if current_beat + duration > total_beats:
                duration = total_beats - current_beat

            measure = int(current_beat // self.beats_per_measure)
            beat_in_measure = current_beat % self.beats_per_measure

            chord_durations.append(ChordDuration(
                chord_index=chord_index,
                start_beat=current_beat,
                duration_beats=duration,
                measure=measure,
                beat_in_measure=beat_in_measure,
                tension_level=0.5  # Neutral tension
            ))

            current_beat += duration
            chord_index += 1

        # Calculate statistics
        avg_density = len(chord_durations) / total_measures
        durations_array = [cd.duration_beats for cd in chord_durations]
        density_variance = float(np.var(durations_array)) if len(durations_array) > 1 else 0.0

        return HarmonicRhythmPattern(
            chord_durations=chord_durations,
            total_measures=total_measures,
            beats_per_measure=self.beats_per_measure,
            average_density=avg_density,
            density_variance=density_variance
        )

    def analyze_chord_density(
        self,
        chord_progression: List[any],
        chord_durations_beats: Optional[List[float]] = None
    ) -> Dict[str, any]:
        """
        Analyze chord density of existing progression

        Args:
            chord_progression: List of chords (any format)
            chord_durations_beats: Optional list of durations in beats
                                  If None, assumes equal durations

        Returns:
            Dictionary with density analysis:
                - total_chords: Number of chords
                - total_measures: Estimated measures
                - chords_per_measure: Average chords per measure
                - density_level: Classification (sparse/medium/high)
                - duration_variance: Variance in durations
                - min_duration: Shortest chord duration
                - max_duration: Longest chord duration
        """
        if not chord_progression:
            return {
                "total_chords": 0,
                "total_measures": 0,
                "chords_per_measure": 0,
                "density_level": "empty"
            }

        total_chords = len(chord_progression)

        # Use provided durations or assume equal
        if chord_durations_beats is None:
            # Assume 1 beat per chord as default
            chord_durations_beats = [1.0] * total_chords

        total_beats = sum(chord_durations_beats)
        total_measures = total_beats / self.beats_per_measure
        chords_per_measure = total_chords / total_measures if total_measures > 0 else 0

        # Classify density
        if chords_per_measure < 0.5:
            density_level = "very_sparse"
        elif chords_per_measure < 1.0:
            density_level = "sparse"
        elif chords_per_measure < 2.0:
            density_level = "low"
        elif chords_per_measure < 4.0:
            density_level = "medium"
        elif chords_per_measure < 8.0:
            density_level = "high"
        else:
            density_level = "very_high"

        duration_variance = float(np.var(chord_durations_beats))
        min_duration = min(chord_durations_beats)
        max_duration = max(chord_durations_beats)

        return {
            "total_chords": total_chords,
            "total_measures": total_measures,
            "total_beats": total_beats,
            "chords_per_measure": chords_per_measure,
            "density_level": density_level,
            "duration_variance": duration_variance,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "durations": chord_durations_beats
        }

    def apply_tension_pacing(
        self,
        progression_length: int,
        tension_curve: Union[TensionCurveType, List[float], Callable],
        base_density: float = 2.0,
        density_range: Tuple[float, float] = (0.5, 4.0)
    ) -> HarmonicRhythmPattern:
        """
        Apply tension-based pacing to harmonic rhythm

        Higher tension = more frequent chord changes (higher density)
        Lower tension = fewer chord changes (lower density)

        Args:
            progression_length: Number of measures
            tension_curve: Tension curve type, list of values, or function
            base_density: Base density (chords per measure)
            density_range: (min_density, max_density) range

        Returns:
            HarmonicRhythmPattern with tension-modulated densities
        """
        # Generate tension values for each measure
        if isinstance(tension_curve, TensionCurveType):
            tension_values = self._generate_tension_curve(
                tension_curve, progression_length
            )
        elif callable(tension_curve):
            tension_values = [tension_curve(i / progression_length)
                            for i in range(progression_length)]
        else:
            # Use provided list, interpolate if needed
            if len(tension_curve) != progression_length:
                tension_values = np.interp(
                    np.linspace(0, len(tension_curve)-1, progression_length),
                    np.arange(len(tension_curve)),
                    tension_curve
                ).tolist()
            else:
                tension_values = list(tension_curve)

        # Normalize tension to 0-1
        min_t, max_t = min(tension_values), max(tension_values)
        if max_t > min_t:
            tension_values = [(t - min_t) / (max_t - min_t)
                            for t in tension_values]

        # Map tension to density
        min_density, max_density = density_range
        chord_durations = []
        current_beat = 0.0
        chord_index = 0

        for measure_idx in range(progression_length):
            tension = tension_values[measure_idx]
            # Higher tension = higher density
            measure_density = min_density + tension * (max_density - min_density)
            chords_in_measure = max(1, int(round(measure_density)))

            # Distribute chords evenly in this measure
            beat_duration = self.beats_per_measure / chords_in_measure

            for i in range(chords_in_measure):
                beat_in_measure = i * beat_duration

                chord_durations.append(ChordDuration(
                    chord_index=chord_index,
                    start_beat=current_beat + beat_in_measure,
                    duration_beats=beat_duration,
                    measure=measure_idx,
                    beat_in_measure=beat_in_measure,
                    tension_level=tension
                ))
                chord_index += 1

            current_beat += self.beats_per_measure

        avg_density = len(chord_durations) / progression_length
        durations = [cd.duration_beats for cd in chord_durations]
        variance = float(np.var(durations))

        return HarmonicRhythmPattern(
            chord_durations=chord_durations,
            total_measures=progression_length,
            beats_per_measure=self.beats_per_measure,
            average_density=avg_density,
            density_variance=variance,
            tension_curve=tension_values
        )

    def _generate_tension_curve(
        self,
        curve_type: TensionCurveType,
        length: int
    ) -> List[float]:
        """Generate tension curve values"""
        x = np.linspace(0, 1, length)

        # Helper to convert to list (handles both numpy arrays and lists)
        def to_list(val):
            if isinstance(val, list):
                return val
            elif hasattr(val, 'tolist'):
                return val.tolist()
            else:
                return list(val)

        if curve_type == TensionCurveType.LINEAR_INCREASE:
            return to_list(x)
        elif curve_type == TensionCurveType.LINEAR_DECREASE:
            if isinstance(x, list):
                return [1 - xi for xi in x]
            return to_list(1 - x)
        elif curve_type == TensionCurveType.EXPONENTIAL_INCREASE:
            if isinstance(x, list):
                return [xi ** 2 for xi in x]
            return to_list(x ** 2)
        elif curve_type == TensionCurveType.EXPONENTIAL_DECREASE:
            if isinstance(x, list):
                return [(1 - xi) ** 2 for xi in x]
            return to_list((1 - x) ** 2)
        elif curve_type == TensionCurveType.ARCH:
            # Rise to peak at midpoint, then fall
            if isinstance(x, list):
                return [4 * xi * (1 - xi) for xi in x]
            return to_list(4 * x * (1 - x))
        elif curve_type == TensionCurveType.VALLEY:
            # Fall to valley at midpoint, then rise
            if isinstance(x, list):
                return [1 - 4 * xi * (1 - xi) for xi in x]
            return to_list(1 - 4 * x * (1 - x))
        elif curve_type == TensionCurveType.WAVE:
            sin_vals = np.sin([xi * 2 * np.pi for xi in x]) if isinstance(x, list) else np.sin(x * 2 * np.pi)
            if isinstance(sin_vals, list):
                return [(s + 1) / 2 for s in sin_vals]
            return to_list((sin_vals + 1) / 2)
        elif curve_type == TensionCurveType.CADENTIAL:
            # Low tension, then accelerate dramatically at end
            steady_part = np.full(int(length * 0.75), 0.3)
            ramp_part = np.linspace(0.3, 1.0, int(length * 0.25))
            combined = np.concatenate([steady_part, ramp_part])
            return to_list(combined)
        else:
            return [0.5] * length  # Neutral

    def create_genre_appropriate_rhythm(
        self,
        genre: Union[str, GenreStyle],
        measures: int = 8,
        variation: float = 0.15
    ) -> HarmonicRhythmPattern:
        """
        Create genre-specific harmonic rhythm

        Based on research into typical harmonic rhythm patterns:
        - Pop: 1-2 bars per chord (0.5-1 chords/measure)
        - Jazz bebop: 2+ chords per bar (2-4 chords/measure)
        - Classical: varies by period
        - Electronic: very sparse (1 per 4-8 bars)

        Args:
            genre: Genre style (string or GenreStyle enum)
            measures: Number of measures
            variation: Variation in durations (0.0-1.0)

        Returns:
            Genre-appropriate HarmonicRhythmPattern
        """
        # Convert string to GenreStyle
        if isinstance(genre, str):
            try:
                genre_enum = GenreStyle[genre.upper().replace("-", "_")]
            except KeyError:
                # Default to medium if unknown genre
                genre_enum = None
                density_range = (1.0, 2.0)
        else:
            genre_enum = genre

        # Get density range for genre
        if genre_enum and genre_enum in self.genre_densities:
            density_range = self.genre_densities[genre_enum]
        else:
            density_range = (1.0, 2.0)  # Default medium

        # Use average of range with variation
        avg_density = (density_range[0] + density_range[1]) / 2

        # For genres with more variation (romantic, etc.), increase variation
        if genre_enum == GenreStyle.CLASSICAL_ROMANTIC:
            variation = max(variation, 0.3)
        elif genre_enum in [GenreStyle.BEBOP, GenreStyle.GOSPEL]:
            # More consistent high density
            variation = min(variation, 0.1)

        pattern = self.generate_harmonic_rhythm(
            density=avg_density,
            total_measures=measures,
            variation=variation,
            prefer_downbeats=(genre_enum != GenreStyle.BEBOP)
        )

        pattern.genre_style = genre_enum
        return pattern

    def apply_harmonic_acceleration(
        self,
        progression_length: int,
        start_density: float = 1.0,
        end_density: float = 4.0,
        curve: str = "exponential",
        cadence_measures: int = 2
    ) -> HarmonicRhythmPattern:
        """
        Apply harmonic acceleration (increase chord change frequency)

        Classical technique: harmonic rhythm accelerates approaching cadences,
        creating forward momentum and emphasizing arrival.

        Args:
            progression_length: Total measures
            start_density: Starting density (chords/measure)
            end_density: Ending density (chords/measure)
            curve: "linear" or "exponential" acceleration
            cadence_measures: Number of measures for cadence acceleration

        Returns:
            HarmonicRhythmPattern with acceleration
        """
        densities = []

        # Calculate densities per measure
        for i in range(progression_length):
            # Determine if in cadence zone
            measures_from_end = progression_length - i

            if measures_from_end <= cadence_measures:
                # In cadence zone - accelerate
                progress = 1 - (measures_from_end / cadence_measures)
                if curve == "exponential":
                    progress = progress ** 2
                density = start_density + progress * (end_density - start_density)
            else:
                # Before cadence - use start density
                density = start_density

            densities.append(density)

        # Generate pattern with varying densities
        chord_durations = []
        current_beat = 0.0
        chord_index = 0

        for measure_idx, density in enumerate(densities):
            chords_in_measure = max(1, int(round(density)))
            beat_duration = self.beats_per_measure / chords_in_measure

            for i in range(chords_in_measure):
                beat_in_measure = i * beat_duration

                # Tension increases with density
                tension = (density - start_density) / (end_density - start_density)
                tension = max(0.0, min(1.0, tension))

                chord_durations.append(ChordDuration(
                    chord_index=chord_index,
                    start_beat=current_beat + beat_in_measure,
                    duration_beats=beat_duration,
                    measure=measure_idx,
                    beat_in_measure=beat_in_measure,
                    tension_level=tension
                ))
                chord_index += 1

            current_beat += self.beats_per_measure

        avg_density = len(chord_durations) / progression_length
        durations = [cd.duration_beats for cd in chord_durations]
        variance = float(np.var(durations))

        return HarmonicRhythmPattern(
            chord_durations=chord_durations,
            total_measures=progression_length,
            beats_per_measure=self.beats_per_measure,
            average_density=avg_density,
            density_variance=variance,
            tension_curve=densities
        )

    def add_suspensions(
        self,
        progression: HarmonicRhythmPattern,
        suspension_rate: float = 0.3,
        anticipation_rate: float = 0.2
    ) -> HarmonicRhythmPattern:
        """
        Add suspensions and anticipations to harmonic rhythm

        Suspensions: Delay chord change past the beat
        Anticipations: Change chord before the beat

        Args:
            progression: Existing HarmonicRhythmPattern
            suspension_rate: Probability of suspension (0.0-1.0)
            anticipation_rate: Probability of anticipation (0.0-1.0)

        Returns:
            Modified HarmonicRhythmPattern with suspensions/anticipations
        """
        new_durations = []

        for i, chord_dur in enumerate(progression.chord_durations):
            new_dur = ChordDuration(
                chord_index=chord_dur.chord_index,
                start_beat=chord_dur.start_beat,
                duration_beats=chord_dur.duration_beats,
                measure=chord_dur.measure,
                beat_in_measure=chord_dur.beat_in_measure,
                tension_level=chord_dur.tension_level,
                is_suspension=chord_dur.is_suspension,
                is_anticipation=chord_dur.is_anticipation
            )

            # Skip first chord
            if i == 0:
                new_durations.append(new_dur)
                continue

            # Randomly apply suspension
            if random.random() < suspension_rate:
                # Delay by 0.25-0.5 beats
                delay = random.uniform(0.25, 0.5)
                new_dur.start_beat += delay
                new_dur.duration_beats -= delay
                new_dur.is_suspension = True

                # Extend previous chord
                new_durations[-1].duration_beats += delay

            # Randomly apply anticipation
            elif random.random() < anticipation_rate and i < len(progression.chord_durations) - 1:
                # Anticipate by 0.25-0.5 beats
                anticipation = random.uniform(0.25, 0.5)
                new_dur.start_beat -= anticipation
                new_dur.duration_beats += anticipation
                new_dur.is_anticipation = True

                # Shorten previous chord
                new_durations[-1].duration_beats -= anticipation

            new_durations.append(new_dur)

        return HarmonicRhythmPattern(
            chord_durations=new_durations,
            total_measures=progression.total_measures,
            beats_per_measure=progression.beats_per_measure,
            average_density=progression.average_density,
            density_variance=progression.density_variance,
            genre_style=progression.genre_style,
            tension_curve=progression.tension_curve
        )

    def apply_harmonic_deceleration(
        self,
        progression_length: int,
        start_density: float = 4.0,
        end_density: float = 1.0,
        curve: str = "exponential"
    ) -> HarmonicRhythmPattern:
        """
        Apply harmonic deceleration (decrease chord change frequency)

        Useful for endings, transitions, or creating calm sections.

        Args:
            progression_length: Total measures
            start_density: Starting density (chords/measure)
            end_density: Ending density (chords/measure)
            curve: "linear" or "exponential" deceleration

        Returns:
            HarmonicRhythmPattern with deceleration
        """
        # Just reverse the acceleration logic
        densities = []

        for i in range(progression_length):
            progress = i / (progression_length - 1) if progression_length > 1 else 0

            if curve == "exponential":
                # Decelerate more dramatically at start
                progress = 1 - (1 - progress) ** 2

            density = start_density + progress * (end_density - start_density)
            densities.append(density)

        # Generate pattern
        chord_durations = []
        current_beat = 0.0
        chord_index = 0

        for measure_idx, density in enumerate(densities):
            chords_in_measure = max(1, int(round(density)))
            beat_duration = self.beats_per_measure / chords_in_measure

            for i in range(chords_in_measure):
                beat_in_measure = i * beat_duration

                # Tension decreases with density for deceleration
                tension = (density - end_density) / (start_density - end_density)
                tension = max(0.0, min(1.0, tension))

                chord_durations.append(ChordDuration(
                    chord_index=chord_index,
                    start_beat=current_beat + beat_in_measure,
                    duration_beats=beat_duration,
                    measure=measure_idx,
                    beat_in_measure=beat_in_measure,
                    tension_level=tension
                ))
                chord_index += 1

            current_beat += self.beats_per_measure

        avg_density = len(chord_durations) / progression_length
        durations = [cd.duration_beats for cd in chord_durations]
        variance = float(np.var(durations))

        return HarmonicRhythmPattern(
            chord_durations=chord_durations,
            total_measures=progression_length,
            beats_per_measure=self.beats_per_measure,
            average_density=avg_density,
            density_variance=variance,
            tension_curve=densities
        )

    def combine_patterns(
        self,
        patterns: List[HarmonicRhythmPattern]
    ) -> HarmonicRhythmPattern:
        """
        Combine multiple harmonic rhythm patterns sequentially

        Args:
            patterns: List of HarmonicRhythmPattern objects

        Returns:
            Combined HarmonicRhythmPattern
        """
        if not patterns:
            raise ValueError("Cannot combine empty pattern list")

        combined_durations = []
        total_measures = 0
        current_beat = 0.0
        chord_index = 0
        all_tension_values = []

        for pattern in patterns:
            for chord_dur in pattern.chord_durations:
                new_measure = total_measures + chord_dur.measure

                combined_durations.append(ChordDuration(
                    chord_index=chord_index,
                    start_beat=current_beat + chord_dur.start_beat -
                               (chord_dur.measure * self.beats_per_measure),
                    duration_beats=chord_dur.duration_beats,
                    measure=new_measure,
                    beat_in_measure=chord_dur.beat_in_measure,
                    is_suspension=chord_dur.is_suspension,
                    is_anticipation=chord_dur.is_anticipation,
                    tension_level=chord_dur.tension_level
                ))
                chord_index += 1

            current_beat += pattern.total_measures * self.beats_per_measure
            total_measures += pattern.total_measures

            if pattern.tension_curve:
                all_tension_values.extend(pattern.tension_curve)

        avg_density = len(combined_durations) / total_measures if total_measures > 0 else 0
        durations = [cd.duration_beats for cd in combined_durations]
        variance = float(np.var(durations)) if len(durations) > 1 else 0.0

        return HarmonicRhythmPattern(
            chord_durations=combined_durations,
            total_measures=total_measures,
            beats_per_measure=self.beats_per_measure,
            average_density=avg_density,
            density_variance=variance,
            tension_curve=all_tension_values if all_tension_values else None
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_harmonic_rhythm_pattern(pattern: HarmonicRhythmPattern) -> None:
    """
    Print human-readable representation of harmonic rhythm pattern

    Args:
        pattern: HarmonicRhythmPattern to display
    """
    print(f"\n{'='*70}")
    print(f"HARMONIC RHYTHM PATTERN")
    print(f"{'='*70}")
    print(f"Total Measures: {pattern.total_measures}")
    print(f"Beats per Measure: {pattern.beats_per_measure}")
    print(f"Average Density: {pattern.average_density:.2f} chords/measure")
    print(f"Density Variance: {pattern.density_variance:.3f}")
    if pattern.genre_style:
        print(f"Genre Style: {pattern.genre_style.value}")
    print(f"\nChord Durations:")
    print(f"{'Chord':<8} {'Measure':<10} {'Beat':<10} {'Duration':<12} {'Tension':<10} {'Flags':<15}")
    print(f"{'-'*70}")

    for cd in pattern.chord_durations:
        flags = []
        if cd.is_suspension:
            flags.append("SUSP")
        if cd.is_anticipation:
            flags.append("ANTIC")
        flags_str = ", ".join(flags) if flags else "-"

        print(f"{cd.chord_index:<8} "
              f"{cd.measure:<10} "
              f"{cd.beat_in_measure:<10.2f} "
              f"{cd.duration_beats:<12.2f} "
              f"{cd.tension_level:<10.2f} "
              f"{flags_str:<15}")

    if pattern.tension_curve:
        print(f"\nTension Curve (per measure):")
        print(", ".join(f"{t:.2f}" for t in pattern.tension_curve))
    print(f"{'='*70}\n")


# ============================================================================
# COMPREHENSIVE UNIT TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HARMONIC RHYTHM MODULE - COMPREHENSIVE TESTS")
    print("=" * 80)

    hr = HarmonicRhythm(beats_per_measure=4)

    # Test 1: Generate medium density rhythm
    print("\n[TEST 1] Generate Medium Density Rhythm (2 chords/bar, 8 measures)")
    pattern1 = hr.generate_harmonic_rhythm(density="medium", total_measures=8)
    assert pattern1.total_measures == 8
    assert len(pattern1.chord_durations) > 0
    assert 1.5 <= pattern1.average_density <= 2.5  # Around 2
    print(f"✓ Generated {len(pattern1.chord_durations)} chords, "
          f"avg density: {pattern1.average_density:.2f}")

    # Test 2: Generate sparse rhythm
    print("\n[TEST 2] Generate Sparse Rhythm (0.5 chords/bar)")
    pattern2 = hr.generate_harmonic_rhythm(density="sparse", total_measures=8)
    assert pattern2.average_density < 1.0
    print(f"✓ Sparse rhythm: {len(pattern2.chord_durations)} chords, "
          f"avg density: {pattern2.average_density:.2f}")

    # Test 3: Generate high density rhythm
    print("\n[TEST 3] Generate High Density Rhythm (4 chords/bar)")
    pattern3 = hr.generate_harmonic_rhythm(density="high", total_measures=4)
    assert pattern3.average_density >= 3.0
    print(f"✓ High density: {len(pattern3.chord_durations)} chords, "
          f"avg density: {pattern3.average_density:.2f}")

    # Test 4: Analyze chord density
    print("\n[TEST 4] Analyze Chord Density")
    test_progression = ["C", "Am", "F", "G", "C", "Am", "Dm", "G"]
    test_durations = [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0]  # beats
    analysis = hr.analyze_chord_density(test_progression, test_durations)
    print(f"✓ Analyzed progression: {analysis['total_chords']} chords, "
          f"{analysis['chords_per_measure']:.2f} per measure, "
          f"density: {analysis['density_level']}")
    assert analysis['total_chords'] == 8
    assert analysis['density_level'] in ["sparse", "low", "medium", "high"]

    # Test 5: Apply linear tension increase
    print("\n[TEST 5] Apply Linear Tension Increase")
    pattern5 = hr.apply_tension_pacing(
        progression_length=8,
        tension_curve=TensionCurveType.LINEAR_INCREASE,
        base_density=2.0,
        density_range=(1.0, 4.0)
    )
    assert len(pattern5.chord_durations) > 0
    # Check that later chords have higher tension
    early_tension = pattern5.chord_durations[2].tension_level
    late_tension = pattern5.chord_durations[-2].tension_level
    assert late_tension > early_tension
    print(f"✓ Linear tension: early={early_tension:.2f}, late={late_tension:.2f}")

    # Test 6: Apply exponential tension increase
    print("\n[TEST 6] Apply Exponential Tension Increase")
    pattern6 = hr.apply_tension_pacing(
        progression_length=8,
        tension_curve=TensionCurveType.EXPONENTIAL_INCREASE,
        density_range=(0.5, 4.0)
    )
    assert len(pattern6.chord_durations) > 0
    print(f"✓ Exponential tension curve applied, "
          f"{len(pattern6.chord_durations)} chords")

    # Test 7: Apply arch tension curve
    print("\n[TEST 7] Apply Arch Tension Curve")
    pattern7 = hr.apply_tension_pacing(
        progression_length=12,
        tension_curve=TensionCurveType.ARCH,
        density_range=(1.0, 3.0)
    )
    # Middle should have higher tension than ends
    if len(pattern7.tension_curve) > 0:
        mid_idx = len(pattern7.tension_curve) // 2
        mid_tension = pattern7.tension_curve[mid_idx]
        start_tension = pattern7.tension_curve[0]
        end_tension = pattern7.tension_curve[-1]
        assert mid_tension > start_tension and mid_tension > end_tension
        print(f"✓ Arch curve: start={start_tension:.2f}, "
              f"mid={mid_tension:.2f}, end={end_tension:.2f}")

    # Test 8: Create pop genre rhythm
    print("\n[TEST 8] Create Pop Genre Rhythm")
    pattern8 = hr.create_genre_appropriate_rhythm(genre="pop", measures=8)
    assert pattern8.genre_style == GenreStyle.POP
    assert 0.4 <= pattern8.average_density <= 1.5  # Pop is sparse
    print(f"✓ Pop rhythm: {pattern8.average_density:.2f} chords/measure")

    # Test 9: Create bebop genre rhythm
    print("\n[TEST 9] Create Bebop Genre Rhythm")
    pattern9 = hr.create_genre_appropriate_rhythm(genre="bebop", measures=8)
    assert pattern9.genre_style == GenreStyle.BEBOP
    assert pattern9.average_density >= 1.8  # Bebop is dense
    print(f"✓ Bebop rhythm: {pattern9.average_density:.2f} chords/measure")

    # Test 10: Create electronic genre rhythm
    print("\n[TEST 10] Create Electronic Genre Rhythm")
    pattern10 = hr.create_genre_appropriate_rhythm(genre="electronic", measures=16)
    assert pattern10.genre_style == GenreStyle.ELECTRONIC
    assert pattern10.average_density < 0.5  # Very sparse
    print(f"✓ Electronic rhythm: {pattern10.average_density:.2f} chords/measure")

    # Test 11: Apply harmonic acceleration
    print("\n[TEST 11] Apply Harmonic Acceleration (1→4 chords/bar)")
    pattern11 = hr.apply_harmonic_acceleration(
        progression_length=12,
        start_density=1.0,
        end_density=4.0,
        cadence_measures=3
    )
    # Later measures should have shorter chord durations (more chords per measure)
    early_measures = [cd for cd in pattern11.chord_durations if cd.measure < 9]
    late_measures = [cd for cd in pattern11.chord_durations if cd.measure >= 9]
    avg_early_duration = sum(cd.duration_beats for cd in early_measures) / len(early_measures) if early_measures else 0
    avg_late_duration = sum(cd.duration_beats for cd in late_measures) / len(late_measures) if late_measures else 0
    print(f"✓ Acceleration: {len(late_measures)} chords in last 3 bars, "
          f"avg early duration: {avg_early_duration:.2f}, avg late duration: {avg_late_duration:.2f}")
    assert avg_late_duration < avg_early_duration  # Shorter durations = faster changes

    # Test 12: Apply exponential acceleration
    print("\n[TEST 12] Apply Exponential Acceleration")
    pattern12 = hr.apply_harmonic_acceleration(
        progression_length=8,
        start_density=1.0,
        end_density=4.0,
        curve="exponential"
    )
    assert len(pattern12.chord_durations) > 8
    print(f"✓ Exponential acceleration: {len(pattern12.chord_durations)} total chords")

    # Test 13: Add suspensions
    print("\n[TEST 13] Add Suspensions to Pattern")
    base_pattern = hr.generate_harmonic_rhythm(density=2.0, total_measures=8)
    pattern13 = hr.add_suspensions(base_pattern, suspension_rate=0.5)
    suspended = sum(1 for cd in pattern13.chord_durations if cd.is_suspension)
    print(f"✓ Added {suspended} suspensions out of {len(pattern13.chord_durations)} chords")
    assert suspended > 0  # Should have some suspensions

    # Test 14: Add anticipations
    print("\n[TEST 14] Add Anticipations to Pattern")
    pattern14 = hr.add_suspensions(
        base_pattern,
        suspension_rate=0.0,
        anticipation_rate=0.5
    )
    anticipated = sum(1 for cd in pattern14.chord_durations if cd.is_anticipation)
    print(f"✓ Added {anticipated} anticipations")

    # Test 15: Apply harmonic deceleration
    print("\n[TEST 15] Apply Harmonic Deceleration (4→1 chords/bar)")
    pattern15 = hr.apply_harmonic_deceleration(
        progression_length=8,
        start_density=4.0,
        end_density=1.0,
        curve="linear"
    )
    # Early measures should have more chords
    early_chords = [cd for cd in pattern15.chord_durations if cd.measure < 3]
    late_chords = [cd for cd in pattern15.chord_durations if cd.measure >= 6]
    print(f"✓ Deceleration: {len(early_chords)} early chords, "
          f"{len(late_chords)} late chords")
    assert len(early_chords) >= len(late_chords)

    # Test 16: Combine patterns
    print("\n[TEST 16] Combine Multiple Patterns")
    p1 = hr.generate_harmonic_rhythm(density=1.0, total_measures=4)
    p2 = hr.generate_harmonic_rhythm(density=2.0, total_measures=4)
    p3 = hr.generate_harmonic_rhythm(density=4.0, total_measures=4)
    combined = hr.combine_patterns([p1, p2, p3])
    assert combined.total_measures == 12
    assert len(combined.chord_durations) == (len(p1.chord_durations) +
                                             len(p2.chord_durations) +
                                             len(p3.chord_durations))
    print(f"✓ Combined 3 patterns: {combined.total_measures} measures, "
          f"{len(combined.chord_durations)} total chords")

    # Test 17: Cadential tension curve
    print("\n[TEST 17] Apply Cadential Tension Curve")
    pattern17 = hr.apply_tension_pacing(
        progression_length=8,
        tension_curve=TensionCurveType.CADENTIAL,
        density_range=(1.0, 4.0)
    )
    # Last measures should have higher tension
    if pattern17.tension_curve:
        assert pattern17.tension_curve[-1] > pattern17.tension_curve[0]
        print(f"✓ Cadential curve: start={pattern17.tension_curve[0]:.2f}, "
              f"end={pattern17.tension_curve[-1]:.2f}")

    # Test 18: Custom tension function
    print("\n[TEST 18] Apply Custom Tension Function")
    custom_fn = lambda x: x ** 3  # Cubic growth
    pattern18 = hr.apply_tension_pacing(
        progression_length=8,
        tension_curve=custom_fn,
        density_range=(1.0, 4.0)
    )
    assert len(pattern18.chord_durations) > 0
    print(f"✓ Custom cubic tension function applied")

    # Test 19: Blues genre rhythm
    print("\n[TEST 19] Create Blues Genre Rhythm")
    pattern19 = hr.create_genre_appropriate_rhythm(genre="blues", measures=12)
    assert pattern19.genre_style == GenreStyle.BLUES
    print(f"✓ Blues rhythm (12-bar): {pattern19.average_density:.2f} chords/measure")

    # Test 20: Verify all durations sum correctly
    print("\n[TEST 20] Verify Total Duration Consistency")
    test_pattern = hr.generate_harmonic_rhythm(density=2.0, total_measures=8)
    total_beats_from_durations = sum(cd.duration_beats
                                     for cd in test_pattern.chord_durations)
    expected_beats = test_pattern.total_measures * hr.beats_per_measure
    # Allow small floating point error
    assert abs(total_beats_from_durations - expected_beats) < 0.1
    print(f"✓ Duration consistency: {total_beats_from_durations:.2f} beats = "
          f"{expected_beats} expected beats")

    # Test 21: Print detailed pattern
    print("\n[TEST 21] Print Detailed Pattern Display")
    demo_pattern = hr.apply_harmonic_acceleration(
        progression_length=4,
        start_density=1.0,
        end_density=4.0
    )
    demo_pattern = hr.add_suspensions(demo_pattern, suspension_rate=0.3)
    print_harmonic_rhythm_pattern(demo_pattern)
    print("✓ Pattern display completed")

    print("\n" + "=" * 80)
    print("ALL 21 TESTS PASSED SUCCESSFULLY!")
    print("=" * 80)
    print("\nModule ready for integration with:")
    print("  - Chord progression generators")
    print("  - MIDI export systems")
    print("  - Harmonic analysis tools")
    print("  - Real-time composition engines")
