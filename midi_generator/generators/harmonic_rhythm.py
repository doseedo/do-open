#!/usr/bin/env python3
"""
Harmonic Rhythm Engine
======================

Controls the timing and density of harmonic changes in musical progressions.
Expands static chord sequences into rhythmically varied harmonic patterns.

Based on:
- Jazz arranging principles (varying chord density for musical interest)
- Bebop harmonic rhythm (fast-moving changes)
- Ballad harmonic rhythm (slow, sustained changes)
- Form-based harmonic planning (intro/verse/chorus/bridge)

Features:
---------
- Flexible chords-per-bar control (0.5, 1, 2, 4 chords/bar)
- Rhythm patterns (standard, fast, slow, mixed, bebop)
- Form-aware harmonic planning
- Anticipation and syncopation
- Held chords over bar lines
- Dynamic harmonic density curves

Author: Agent 4 - Harmonic Progression Designer
Date: 2025
License: MIT
"""

import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from genres.jazz import JazzChord
from parameters import registry, param, ParameterType, MusicalDomain


@dataclass
class ChordEvent:
    """
    Chord with specific timing information.

    Attributes:
        chord: The JazzChord object
        start_time: Start time in beats
        duration: Duration in beats
        anticipation: Anticipate by this many beats (play early)
    """
    chord: JazzChord
    start_time: float
    duration: float
    anticipation: float = 0.0


class RhythmPattern(Enum):
    """Harmonic rhythm patterns"""
    STANDARD = "standard"        # 1 chord per bar
    FAST = "fast"                # 2 chords per bar (ii-V, I-IV, etc.)
    SLOW = "slow"                # 1 chord per 2 bars (ballads)
    MIXED = "mixed"              # Varying (bebop style)
    BEBOP = "bebop"              # Complex, frequent changes
    LATIN = "latin"              # 2-bar patterns (montuno, clave)
    MODAL = "modal"              # Very slow, static


class HarmonicRhythmEngine:
    """
    Engine for controlling harmonic rhythm and chord timing.

    Transforms static chord progressions into time-based chord events
    with varied rhythmic patterns.
    """

    # Register parameters on module load
    _params_registered = False

    def __init__(self, **params):
        """
        Initialize harmonic rhythm engine.

        Args:
            **params: Parameter overrides
        """
        self.params = params
        self._register_parameters()

    @classmethod
    def _register_parameters(cls):
        """Register all harmonic rhythm parameters in global registry"""
        if cls._params_registered:
            return

        # Timing parameters
        registry.register_parameter(
            name="rhythm.timing.beats_per_bar",
            type=ParameterType.INTEGER,
            default=4,
            description="Beats per bar (time signature numerator)",
            range=(2, 12),
            domain=MusicalDomain.RHYTHM,
            module="harmonic_rhythm",
            musical_impact="high",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.anticipation.eighth_note",
            type=ParameterType.CONTINUOUS,
            default=0.125,
            description="Eighth note anticipation amount in beats",
            range=(0.0, 0.5),
            domain=MusicalDomain.RHYTHM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "bebop", "swing"]
        )

        # Duration multipliers for different styles
        registry.register_parameter(
            name="rhythm.ballad.duration_multiplier",
            type=ParameterType.CONTINUOUS,
            default=2.0,
            description="Duration multiplier for ballad style (chords held longer)",
            range=(1.5, 4.0),
            domain=MusicalDomain.RHYTHM,
            module="harmonic_rhythm",
            musical_impact="high",
            genre_relevance=["jazz", "ballad"]
        )

        registry.register_parameter(
            name="rhythm.modal.duration_multiplier",
            type=ParameterType.CONTINUOUS,
            default=4.0,
            description="Duration multiplier for modal jazz (very slow changes)",
            range=(2.0, 8.0),
            domain=MusicalDomain.RHYTHM,
            module="harmonic_rhythm",
            musical_impact="high",
            genre_relevance=["jazz", "modal", "fusion"]
        )

        # Bebop rhythm patterns
        registry.register_parameter(
            name="rhythm.bebop.patterns",
            type=ParameterType.ARRAY,
            default=[4.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0],
            description="Bebop duration patterns in beats",
            domain=MusicalDomain.RHYTHM,
            module="harmonic_rhythm",
            musical_impact="high",
            genre_relevance=["jazz", "bebop"]
        )

        # Default rhythm patterns for form sections
        registry.register_parameter(
            name="rhythm.form.intro_pattern",
            type=ParameterType.CATEGORICAL,
            default="slow",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for intro sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.form.verse_pattern",
            type=ParameterType.CATEGORICAL,
            default="standard",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for verse sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.form.chorus_pattern",
            type=ParameterType.CATEGORICAL,
            default="fast",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for chorus sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.form.bridge_pattern",
            type=ParameterType.CATEGORICAL,
            default="mixed",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for bridge sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.form.outro_pattern",
            type=ParameterType.CATEGORICAL,
            default="slow",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for outro sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        registry.register_parameter(
            name="rhythm.form.solo_pattern",
            type=ParameterType.CATEGORICAL,
            default="bebop",
            options=["slow", "standard", "fast", "mixed", "bebop", "latin", "modal"],
            description="Default rhythm pattern for solo sections",
            domain=MusicalDomain.FORM,
            module="harmonic_rhythm",
            musical_impact="medium",
            genre_relevance=["jazz", "all"]
        )

        cls._params_registered = True

    # ========================================================================
    # MAIN EXPANSION FUNCTION
    # ========================================================================

    def expand_progression(
        self,
        base_progression: List[JazzChord],
        bars: int,
        chords_per_bar: float = 1.0,
        rhythm_pattern: str = "standard",
        use_anticipation: bool = False,
        **kwargs
    ) -> List[ChordEvent]:
        """
        Create chord events with specific timing from a base progression.

        Args:
            base_progression: Base chord progression (no timing)
            bars: Total number of bars to generate
            chords_per_bar: Average chords per bar (0.5, 1, 2, 4)
            rhythm_pattern: Rhythm pattern to apply
            use_anticipation: Add chord anticipations (jazz style)
            **kwargs: Parameter overrides

        Returns:
            List of ChordEvent objects with specific timing
        """
        # Merge instance params with method params
        params = {**self.params, **kwargs}

        chord_events = []
        current_beat = 0.0
        beats_per_bar = param("rhythm.timing.beats_per_bar", params, 4.0)

        # Calculate how to distribute chords
        total_beats = bars * beats_per_bar
        num_chords = len(base_progression)

        # Get parameter values
        anticipation_amount = param("rhythm.anticipation.eighth_note", params, 0.125)
        ballad_multiplier = param("rhythm.ballad.duration_multiplier", params, 2.0)
        modal_multiplier = param("rhythm.modal.duration_multiplier", params, 4.0)

        if rhythm_pattern == "standard":
            # One chord per bar
            beats_per_chord = beats_per_bar
            for chord in base_progression:
                duration = beats_per_chord
                anticipation = anticipation_amount if use_anticipation else 0.0

                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=anticipation
                )
                chord_events.append(event)
                current_beat += duration

        elif rhythm_pattern == "fast":
            # Two chords per bar (common in bebop)
            beats_per_chord = beats_per_bar / 2.0
            for i, chord in enumerate(base_progression):
                duration = beats_per_chord
                anticipation = anticipation_amount if use_anticipation and i % 2 == 1 else 0.0

                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=anticipation
                )
                chord_events.append(event)
                current_beat += duration

        elif rhythm_pattern == "slow":
            # One chord every 2 bars (ballad style)
            beats_per_chord = beats_per_bar * ballad_multiplier
            for chord in base_progression:
                duration = beats_per_chord
                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=0.0  # No anticipation in ballads
                )
                chord_events.append(event)
                current_beat += duration

        elif rhythm_pattern == "mixed" or rhythm_pattern == "bebop":
            # Varying chord durations (bebop style)
            durations = self._generate_bebop_durations(
                num_chords,
                total_beats,
                params
            )

            for i, chord in enumerate(base_progression):
                duration = durations[i] if i < len(durations) else beats_per_bar
                # Anticipate dominant chords in bebop
                anticipation = 0.0
                if use_anticipation and chord.quality == "dom7":
                    anticipation = anticipation_amount

                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=anticipation
                )
                chord_events.append(event)
                current_beat += duration

        elif rhythm_pattern == "latin":
            # 2-bar patterns (clave-based)
            pattern = [beats_per_bar, beats_per_bar]  # 2-bar cycle
            pattern_index = 0

            for chord in base_progression:
                duration = pattern[pattern_index % len(pattern)]
                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=0.0
                )
                chord_events.append(event)
                current_beat += duration
                pattern_index += 1

        elif rhythm_pattern == "modal":
            # Very slow harmonic rhythm (modal jazz)
            beats_per_chord = beats_per_bar * modal_multiplier
            for chord in base_progression:
                duration = beats_per_chord
                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=duration,
                    anticipation=0.0
                )
                chord_events.append(event)
                current_beat += duration

        else:
            # Default: standard rhythm
            beats_per_chord = beats_per_bar
            for chord in base_progression:
                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=beats_per_chord,
                    anticipation=0.0
                )
                chord_events.append(event)
                current_beat += beats_per_chord

        return chord_events

    # ========================================================================
    # BEBOP DURATION GENERATOR
    # ========================================================================

    def _generate_bebop_durations(
        self,
        num_chords: int,
        total_beats: float,
        params: Dict[str, Any]
    ) -> List[float]:
        """
        Generate bebop-style varied chord durations.

        Bebop uses a mix of 1-bar, 2-bar, and occasional half-bar changes.

        Args:
            num_chords: Number of chords to distribute
            total_beats: Total beats available
            params: Parameter overrides

        Returns:
            List of durations in beats
        """
        durations = []
        remaining_beats = total_beats
        chords_left = num_chords

        # Common bebop duration patterns (parameterized)
        patterns = param("rhythm.bebop.patterns", params, [4.0, 4.0, 2.0, 4.0, 8.0, 4.0, 2.0, 2.0])

        for i in range(num_chords):
            if chords_left == 1:
                # Last chord gets all remaining beats
                durations.append(remaining_beats)
                break

            # Choose duration from pattern
            duration = patterns[i % len(patterns)]

            # Ensure we don't exceed remaining beats
            if duration > remaining_beats - (chords_left - 1) * 2.0:
                # Adjust to fit
                duration = max(2.0, remaining_beats - (chords_left - 1) * 2.0)

            durations.append(duration)
            remaining_beats -= duration
            chords_left -= 1

        return durations

    # ========================================================================
    # FORM-BASED HARMONIC RHYTHM
    # ========================================================================

    def create_form_based_rhythm(
        self,
        progressions: Dict[str, List[JazzChord]],
        form_structure: Dict[str, int],
        rhythm_map: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> Dict[str, List[ChordEvent]]:
        """
        Create harmonic rhythm based on musical form sections.

        Args:
            progressions: Dict of section name to chord progression
            form_structure: Dict of section name to bar count
            rhythm_map: Dict of section name to rhythm pattern (optional)
            **kwargs: Parameter overrides

        Returns:
            Dict of section name to chord events

        Example:
            progressions = {
                "intro": [Cmaj7, Fmaj7],
                "A": [Dm7, G7, Cmaj7, Am7],
                "B": [Em7, A7, Dm7, G7]
            }
            form_structure = {"intro": 4, "A": 8, "B": 8}
            rhythm_map = {"intro": "slow", "A": "standard", "B": "fast"}
        """
        # Merge instance params with method params
        params = {**self.params, **kwargs}
        result = {}

        # Default rhythm patterns for sections (parameterized)
        default_rhythm_map = {
            "intro": param("rhythm.form.intro_pattern", params, "slow"),
            "verse": param("rhythm.form.verse_pattern", params, "standard"),
            "chorus": param("rhythm.form.chorus_pattern", params, "fast"),
            "bridge": param("rhythm.form.bridge_pattern", params, "mixed"),
            "outro": param("rhythm.form.outro_pattern", params, "slow"),
            "solo": param("rhythm.form.solo_pattern", params, "bebop")
        }

        for section_name, chord_progression in progressions.items():
            bars = form_structure.get(section_name, 8)

            # Determine rhythm pattern
            if rhythm_map and section_name in rhythm_map:
                pattern = rhythm_map[section_name]
            elif section_name in default_rhythm_map:
                pattern = default_rhythm_map[section_name]
            else:
                pattern = "standard"

            # Expand progression with rhythm
            chord_events = self.expand_progression(
                chord_progression,
                bars=bars,
                rhythm_pattern=pattern,
                use_anticipation=True,
                **params
            )

            result[section_name] = chord_events

        return result

    # ========================================================================
    # HARMONIC DENSITY CURVE
    # ========================================================================

    def apply_density_curve(
        self,
        base_progression: List[JazzChord],
        total_bars: int,
        density_curve: List[float],
        **kwargs
    ) -> List[ChordEvent]:
        """
        Apply a harmonic density curve over time.

        Args:
            base_progression: Base chord progression
            total_bars: Total number of bars
            density_curve: List of density values (0.5-4.0) per section
            **kwargs: Parameter overrides

        Returns:
            Chord events with varying harmonic density

        Example:
            density_curve = [1.0, 2.0, 4.0, 2.0, 1.0]
            Creates gradual build-up and release of harmonic activity
        """
        # Merge instance params with method params
        params = {**self.params, **kwargs}

        chord_events = []
        current_beat = 0.0
        beats_per_bar = param("rhythm.timing.beats_per_bar", params, 4.0)

        # Divide progression into sections based on density curve
        num_sections = len(density_curve)
        chords_per_section = len(base_progression) // num_sections
        bars_per_section = total_bars // num_sections

        section_idx = 0
        chord_idx = 0

        for section in range(num_sections):
            section_density = density_curve[section]
            section_bars = bars_per_section
            section_beats = section_bars * beats_per_bar

            # Get chords for this section
            start_idx = section * chords_per_section
            end_idx = start_idx + chords_per_section
            if section == num_sections - 1:
                # Last section gets remaining chords
                end_idx = len(base_progression)
            section_chords = base_progression[start_idx:end_idx]

            # Calculate duration based on density
            beats_per_chord = section_beats / max(len(section_chords), 1)

            # Create events for this section
            for chord in section_chords:
                event = ChordEvent(
                    chord=chord,
                    start_time=current_beat,
                    duration=beats_per_chord,
                    anticipation=0.0
                )
                chord_events.append(event)
                current_beat += beats_per_chord

        return chord_events

    # ========================================================================
    # UTILITY FUNCTIONS
    # ========================================================================

    def calculate_harmonic_rhythm_rate(
        self,
        chord_events: List[ChordEvent],
        **kwargs
    ) -> float:
        """
        Calculate average harmonic rhythm rate (chords per bar).

        Args:
            chord_events: List of chord events
            **kwargs: Parameter overrides

        Returns:
            Average chords per bar
        """
        if not chord_events:
            return 0.0

        # Merge instance params with method params
        params = {**self.params, **kwargs}

        total_duration = sum(event.duration for event in chord_events)
        total_chords = len(chord_events)
        beats_per_bar = param("rhythm.timing.beats_per_bar", params, 4.0)

        total_bars = total_duration / beats_per_bar
        return total_chords / max(total_bars, 1.0)

    def analyze_harmonic_rhythm(
        self,
        chord_events: List[ChordEvent],
        **kwargs
    ) -> Dict:
        """
        Analyze harmonic rhythm characteristics.

        Args:
            chord_events: List of chord events
            **kwargs: Parameter overrides

        Returns:
            Dictionary with analysis metrics
        """
        if not chord_events:
            return {}

        durations = [event.duration for event in chord_events]
        anticipations = [event.anticipation for event in chord_events]

        return {
            "total_chords": len(chord_events),
            "total_duration": sum(durations),
            "avg_chord_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "anticipation_count": sum(1 for a in anticipations if a > 0),
            "anticipation_ratio": sum(1 for a in anticipations if a > 0) / len(anticipations),
            "harmonic_rhythm_rate": self.calculate_harmonic_rhythm_rate(chord_events, **kwargs)
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("HARMONIC RHYTHM ENGINE - EXAMPLES")
    print("=" * 80)

    # Create a simple progression
    from genres.jazz import JazzChord

    progression = [
        JazzChord(root=2, quality="min7"),   # Dm7
        JazzChord(root=7, quality="dom7"),   # G7
        JazzChord(root=0, quality="maj7"),   # Cmaj7
        JazzChord(root=9, quality="min7"),   # Am7
    ]

    engine = HarmonicRhythmEngine()

    print("\nBase progression (no timing):")
    for i, chord in enumerate(progression, 1):
        print(f"  {i}. {chord}")

    # Example 1: Standard rhythm (1 chord/bar)
    print("\n" + "-" * 80)
    print("1. STANDARD RHYTHM (1 chord per bar)")
    print("-" * 80)
    standard = engine.expand_progression(
        progression,
        bars=4,
        rhythm_pattern="standard"
    )
    for event in standard:
        print(f"  Beat {event.start_time:5.1f}: {event.chord} (duration={event.duration} beats)")

    # Example 2: Fast rhythm (2 chords/bar)
    print("\n" + "-" * 80)
    print("2. FAST RHYTHM (2 chords per bar)")
    print("-" * 80)
    fast = engine.expand_progression(
        progression,
        bars=2,
        rhythm_pattern="fast",
        use_anticipation=True
    )
    for event in fast:
        ant_str = f", anticipation={event.anticipation}" if event.anticipation > 0 else ""
        print(f"  Beat {event.start_time:5.1f}: {event.chord} (duration={event.duration}{ant_str})")

    # Example 3: Bebop mixed rhythm
    print("\n" + "-" * 80)
    print("3. BEBOP MIXED RHYTHM (varying durations)")
    print("-" * 80)
    bebop = engine.expand_progression(
        progression * 2,  # 8 chords
        bars=16,
        rhythm_pattern="bebop",
        use_anticipation=True
    )
    for event in bebop:
        ant_str = f", anticipation={event.anticipation}" if event.anticipation > 0 else ""
        print(f"  Beat {event.start_time:5.1f}: {event.chord} (duration={event.duration}{ant_str})")

    # Example 4: Form-based rhythm
    print("\n" + "-" * 80)
    print("4. FORM-BASED RHYTHM (intro/A/B sections)")
    print("-" * 80)

    form_progressions = {
        "intro": [
            JazzChord(root=0, quality="maj7"),
            JazzChord(root=5, quality="maj7"),
        ],
        "A": progression,
        "B": [
            JazzChord(root=4, quality="min7"),
            JazzChord(root=9, quality="dom7"),
            JazzChord(root=2, quality="min7"),
            JazzChord(root=7, quality="dom7"),
        ]
    }

    form_structure = {
        "intro": 4,
        "A": 8,
        "B": 8
    }

    form_rhythm = engine.create_form_based_rhythm(
        form_progressions,
        form_structure
    )

    for section_name, events in form_rhythm.items():
        print(f"\n  Section: {section_name}")
        for event in events:
            print(f"    Beat {event.start_time:5.1f}: {event.chord} (duration={event.duration})")

    # Example 5: Density curve
    print("\n" + "-" * 80)
    print("5. HARMONIC DENSITY CURVE (build and release)")
    print("-" * 80)

    long_progression = progression * 4  # 16 chords
    density_curve = [1.0, 2.0, 3.0, 2.0, 1.0]  # Build to middle, then release

    density_events = engine.apply_density_curve(
        long_progression,
        total_bars=20,
        density_curve=density_curve
    )

    print(f"  Density curve: {density_curve}")
    print(f"  Total events: {len(density_events)}")
    print("\n  Sample events:")
    for i, event in enumerate(density_events[:8], 1):
        print(f"    {i}. Beat {event.start_time:5.1f}: {event.chord} (duration={event.duration})")

    # Analyze harmonic rhythm
    print("\n" + "-" * 80)
    print("6. HARMONIC RHYTHM ANALYSIS")
    print("-" * 80)

    analysis = engine.analyze_harmonic_rhythm(bebop)
    print(f"\n  Bebop rhythm analysis:")
    print(f"    Total chords: {analysis['total_chords']}")
    print(f"    Total duration: {analysis['total_duration']} beats")
    print(f"    Average chord duration: {analysis['avg_chord_duration']:.2f} beats")
    print(f"    Duration range: {analysis['min_duration']:.1f} - {analysis['max_duration']:.1f} beats")
    print(f"    Anticipation ratio: {analysis['anticipation_ratio']:.1%}")
    print(f"    Harmonic rhythm rate: {analysis['harmonic_rhythm_rate']:.2f} chords/bar")

    print("\n" + "=" * 80)
