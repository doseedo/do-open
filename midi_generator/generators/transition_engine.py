#!/usr/bin/env python3
"""
Transition Engine - Modulation & Section Transition System
Part of the Ultimate MIDI Generation Library

This module handles smooth transitions between musical sections:
- Modulation techniques (common chord, direct, sequential, enharmonic)
- Section transitions (build-ups, breakdowns, fills)
- Dynamic transitions (crescendo/decrescendo)
- Harmonic transitions (pivot chords, secondary dominants)
- Turnarounds (jazz, blues, classical cadences)

Author: Agent 5 - Form & Structure Engine
Research: Kostka & Payne's Tonal Harmony, Piston's Harmony, Jazz Theory sources
"""

import random
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class ModulationType(Enum):
    """Types of modulation techniques"""
    COMMON_CHORD = "common_chord"  # Pivot chord modulation
    DIRECT = "direct"  # Abrupt key change
    SEQUENTIAL = "sequential"  # Through sequence
    ENHARMONIC = "enharmonic"  # Enharmonic reinterpretation
    CHROMATIC_MEDIANT = "chromatic_mediant"  # Third relation
    MODAL_MIXTURE = "modal_mixture"  # Through borrowed chords


class TransitionType(Enum):
    """Types of section transitions"""
    BUILD_UP = "build_up"  # Increase energy
    BREAKDOWN = "breakdown"  # Reduce texture
    FILL = "fill"  # Drum/melodic fill
    CRESCENDO = "crescendo"  # Dynamic increase
    DECRESCENDO = "decrescendo"  # Dynamic decrease
    RISER = "riser"  # EDM-style riser
    TURNAROUND = "turnaround"  # Harmonic turnaround
    PAUSE = "pause"  # Fermata or GP
    ACCELERANDO = "accelerando"  # Speed up
    RITARDANDO = "ritardando"  # Slow down


@dataclass
class Modulation:
    """
    Represents a modulation from one key to another

    Attributes:
        from_key: Starting key (MIDI note)
        to_key: Destination key (MIDI note)
        from_major: True if starting key is major
        to_major: True if destination key is major
        technique: Modulation technique used
        pivot_chords: List of pivot chords (if applicable)
        length_bars: Number of bars for modulation
        preparation_bars: Bars to prepare the modulation
    """
    from_key: int
    to_key: int
    from_major: bool
    to_major: bool
    technique: ModulationType
    pivot_chords: List[str] = None
    length_bars: int = 4
    preparation_bars: int = 2

    def __post_init__(self):
        if self.pivot_chords is None:
            self.pivot_chords = []


@dataclass
class Transition:
    """
    Represents a transition between sections

    Attributes:
        transition_type: Type of transition
        length_bars: Number of bars
        start_dynamic: Starting dynamic level (0-1)
        end_dynamic: Ending dynamic level (0-1)
        start_texture: Starting texture density (0-1)
        end_texture: Ending texture density (0-1)
        rhythmic_pattern: Rhythmic characteristics
        harmonic_content: Chord progression for transition
    """
    transition_type: TransitionType
    length_bars: int
    start_dynamic: float = 0.5
    end_dynamic: float = 0.5
    start_texture: float = 0.5
    end_texture: float = 0.5
    rhythmic_pattern: str = "steady"
    harmonic_content: List[str] = None

    def __post_init__(self):
        if self.harmonic_content is None:
            self.harmonic_content = []


# ============================================================================
# SCALE & CHORD UTILITIES
# ============================================================================

class ScaleUtility:
    """Utility for scale and chord calculations"""

    # Diatonic chords in major and minor keys
    MAJOR_SCALE_CHORDS = ['I', 'ii', 'iii', 'IV', 'V', 'vi', 'vii°']
    MINOR_SCALE_CHORDS = ['i', 'ii°', 'III', 'iv', 'v', 'VI', 'VII']

    # Intervals (semitones)
    INTERVALS = {
        'unison': 0,
        'minor_2nd': 1,
        'major_2nd': 2,
        'minor_3rd': 3,
        'major_3rd': 4,
        'perfect_4th': 5,
        'tritone': 6,
        'perfect_5th': 7,
        'minor_6th': 8,
        'major_6th': 9,
        'minor_7th': 10,
        'major_7th': 11,
        'octave': 12
    }

    @staticmethod
    def get_diatonic_chords(key: int, is_major: bool) -> List[Tuple[str, int]]:
        """
        Get diatonic chords for a key

        Args:
            key: Tonic MIDI note
            is_major: True for major, False for minor

        Returns:
            List of (chord_symbol, root_midi_note) tuples
        """
        if is_major:
            scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale
            chord_types = ['maj', 'min', 'min', 'maj', 'maj', 'min', 'dim']
            roman_numerals = ScaleUtility.MAJOR_SCALE_CHORDS
        else:
            scale_intervals = [0, 2, 3, 5, 7, 8, 10]  # Natural minor
            chord_types = ['min', 'dim', 'maj', 'min', 'min', 'maj', 'maj']
            roman_numerals = ScaleUtility.MINOR_SCALE_CHORDS

        chords = []
        for i, (interval, chord_type, numeral) in enumerate(zip(scale_intervals, chord_types, roman_numerals)):
            root_note = key + interval
            chords.append((numeral, root_note, chord_type))

        return chords

    @staticmethod
    def find_common_chords(key1: int, is_major1: bool, key2: int, is_major2: bool) -> List[str]:
        """
        Find common chords between two keys (pivot chords)

        Args:
            key1: First key tonic
            is_major1: True if first key is major
            key2: Second key tonic
            is_major2: True if second key is major

        Returns:
            List of common chord symbols
        """
        chords1 = ScaleUtility.get_diatonic_chords(key1, is_major1)
        chords2 = ScaleUtility.get_diatonic_chords(key2, is_major2)

        # Find chords with same root and type
        common = []
        for numeral1, root1, type1 in chords1:
            for numeral2, root2, type2 in chords2:
                if root1 % 12 == root2 % 12 and type1 == type2:
                    common.append(f"{numeral1}/{numeral2}")

        return common

    @staticmethod
    def get_secondary_dominant(degree: int, key: int, is_major: bool) -> Tuple[int, str]:
        """
        Get secondary dominant for a scale degree

        Args:
            degree: Scale degree (1-7)
            key: Tonic MIDI note
            is_major: True for major

        Returns:
            (root_note, chord_symbol) tuple
        """
        scale_intervals = [0, 2, 4, 5, 7, 9, 11] if is_major else [0, 2, 3, 5, 7, 8, 10]
        target_note = key + scale_intervals[degree - 1]
        dominant_note = target_note + 7  # Perfect 5th above target
        return (dominant_note, "V7")


# ============================================================================
# MODULATION GENERATORS
# ============================================================================

class CommonChordModulation:
    """
    Generate common chord (pivot chord) modulation

    This is the most common modulation technique in classical music.
    Uses a chord that exists in both keys as a pivot.
    """

    @staticmethod
    def generate(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool,
        length_bars: int = 4
    ) -> Modulation:
        """
        Generate a common chord modulation

        Args:
            from_key: Starting key tonic
            from_major: True if starting key is major
            to_key: Destination key tonic
            to_major: True if destination key is major
            length_bars: Number of bars for modulation

        Returns:
            Modulation object with pivot chords
        """
        # Find common chords
        common_chords = ScaleUtility.find_common_chords(
            from_key, from_major, to_key, to_major
        )

        # Select best pivot chord (prefer IV or ii in old key → I or V in new key)
        pivot = common_chords[0] if common_chords else "I/V"

        return Modulation(
            from_key=from_key,
            to_key=to_key,
            from_major=from_major,
            to_major=to_major,
            technique=ModulationType.COMMON_CHORD,
            pivot_chords=[pivot],
            length_bars=length_bars,
            preparation_bars=2
        )


class DirectModulation:
    """
    Direct (abrupt) modulation without preparation

    Used in:
    - Pop music (chorus in different key)
    - Dramatic effect
    - Film music
    """

    @staticmethod
    def generate(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool,
        use_rest: bool = True
    ) -> Modulation:
        """
        Generate direct modulation

        Args:
            from_key: Starting key
            from_major: True if major
            to_key: Destination key
            to_major: True if major
            use_rest: Include brief rest before modulation

        Returns:
            Modulation object
        """
        return Modulation(
            from_key=from_key,
            to_key=to_key,
            from_major=from_major,
            to_major=to_major,
            technique=ModulationType.DIRECT,
            pivot_chords=[],
            length_bars=0 if use_rest else 0,
            preparation_bars=0
        )


class SequentialModulation:
    """
    Modulation through sequence

    The same melodic/harmonic pattern is repeated at different pitch levels,
    gradually moving to the new key.
    """

    @staticmethod
    def generate(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool,
        num_steps: int = 3
    ) -> Modulation:
        """
        Generate sequential modulation

        Args:
            from_key: Starting key
            from_major: True if major
            to_key: Destination key
            to_major: True if major
            num_steps: Number of sequential steps

        Returns:
            Modulation object with sequence
        """
        # Calculate intermediate steps
        interval = to_key - from_key
        step_size = interval // num_steps

        intermediate_chords = []
        for i in range(num_steps):
            transposed_key = from_key + (step_size * i)
            intermediate_chords.append(f"Seq_{i+1}")

        return Modulation(
            from_key=from_key,
            to_key=to_key,
            from_major=from_major,
            to_major=to_major,
            technique=ModulationType.SEQUENTIAL,
            pivot_chords=intermediate_chords,
            length_bars=num_steps * 2,
            preparation_bars=0
        )


class EnharmonicModulation:
    """
    Enharmonic modulation (reinterpretation of chord)

    Uses enharmonic equivalence (e.g., G# = Ab) to pivot to distant keys.
    Common techniques:
    - Diminished 7th chord (can resolve to 8 different keys)
    - Augmented 6th → Dominant 7th reinterpretation
    - German augmented 6th = Dominant 7th
    """

    @staticmethod
    def generate(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool
    ) -> Modulation:
        """
        Generate enharmonic modulation

        Args:
            from_key: Starting key
            from_major: True if major
            to_key: Destination key
            to_major: True if major

        Returns:
            Modulation object using enharmonic pivot
        """
        # Use diminished 7th as common enharmonic pivot
        pivot_chord = "vii°7"

        return Modulation(
            from_key=from_key,
            to_key=to_key,
            from_major=from_major,
            to_major=to_major,
            technique=ModulationType.ENHARMONIC,
            pivot_chords=[pivot_chord, "V7"],
            length_bars=4,
            preparation_bars=2
        )


class ChromaticMediantModulation:
    """
    Chromatic mediant modulation (third relations)

    Modulation to keys a third away:
    - C major → E major (up major 3rd)
    - C major → Ab major (down major 3rd)
    - C major → Eb major (up minor 3rd)

    Creates dreamy, cinematic effect (common in film music)
    """

    @staticmethod
    def generate(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool
    ) -> Modulation:
        """
        Generate chromatic mediant modulation

        Args:
            from_key: Starting key
            from_major: True if major
            to_key: Destination key (should be 3rd or 4th away)
            to_major: True if major

        Returns:
            Modulation object
        """
        # Calculate interval
        interval = abs(to_key - from_key) % 12

        if interval in [3, 4, 8, 9]:  # Third relations
            technique = ModulationType.CHROMATIC_MEDIANT
        else:
            technique = ModulationType.DIRECT

        return Modulation(
            from_key=from_key,
            to_key=to_key,
            from_major=from_major,
            to_major=to_major,
            technique=technique,
            pivot_chords=["I", "I"],  # Direct juxtaposition
            length_bars=2,
            preparation_bars=1
        )


# ============================================================================
# SECTION TRANSITIONS
# ============================================================================

class BuildUpTransition:
    """
    Build-up transition (increase energy toward climax)

    Techniques:
    - Add instruments progressively
    - Increase rhythmic density
    - Crescendo
    - Drum fill at end
    """

    @staticmethod
    def generate(
        length_bars: int = 4,
        intensity: float = 0.8
    ) -> Transition:
        """
        Generate build-up transition

        Args:
            length_bars: Number of bars
            intensity: Final intensity (0-1)

        Returns:
            Transition object
        """
        return Transition(
            transition_type=TransitionType.BUILD_UP,
            length_bars=length_bars,
            start_dynamic=0.4,
            end_dynamic=min(0.4 + (intensity * 0.5), 1.0),
            start_texture=0.3,
            end_texture=min(0.3 + (intensity * 0.6), 1.0),
            rhythmic_pattern="accelerating_density",
            harmonic_content=["I", "IV", "V", "V"]  # Dominant build
        )


class BreakdownTransition:
    """
    Breakdown transition (reduce texture/energy)

    Techniques:
    - Remove instruments
    - Simplify rhythm
    - Decrescendo
    - Filter sweep (EDM)
    """

    @staticmethod
    def generate(
        length_bars: int = 4,
        final_texture: float = 0.2
    ) -> Transition:
        """
        Generate breakdown transition

        Args:
            length_bars: Number of bars
            final_texture: Final texture density (0-1)

        Returns:
            Transition object
        """
        return Transition(
            transition_type=TransitionType.BREAKDOWN,
            length_bars=length_bars,
            start_dynamic=0.7,
            end_dynamic=0.3,
            start_texture=0.8,
            end_texture=final_texture,
            rhythmic_pattern="simplifying",
            harmonic_content=["I", "vi", "IV", "I"]
        )


class DrumFillTransition:
    """
    Drum fill transition (rhythmic transition between sections)

    Types:
    - Linear fill (straight 16ths)
    - Triplet fill
    - Flam fill
    - Paradiddle fill
    """

    @staticmethod
    def generate(
        length_bars: int = 1,
        fill_style: str = "linear"
    ) -> Transition:
        """
        Generate drum fill transition

        Args:
            length_bars: Number of bars (usually 1-2)
            fill_style: Style of fill

        Returns:
            Transition object
        """
        return Transition(
            transition_type=TransitionType.FILL,
            length_bars=length_bars,
            start_dynamic=0.6,
            end_dynamic=0.8,
            start_texture=0.5,
            end_texture=0.7,
            rhythmic_pattern=fill_style,
            harmonic_content=[]  # Drums only
        )


class RiserTransition:
    """
    Riser transition (EDM-style build with pitch rise)

    Techniques:
    - White noise riser
    - Pitch-rising synth
    - Reverse cymbal
    - Filter sweep
    """

    @staticmethod
    def generate(
        length_bars: int = 8,
        start_pitch: int = 48,
        end_pitch: int = 72
    ) -> Transition:
        """
        Generate riser transition

        Args:
            length_bars: Number of bars
            start_pitch: Starting pitch (MIDI)
            end_pitch: Ending pitch (MIDI)

        Returns:
            Transition object
        """
        return Transition(
            transition_type=TransitionType.RISER,
            length_bars=length_bars,
            start_dynamic=0.3,
            end_dynamic=0.9,
            start_texture=0.2,
            end_texture=0.8,
            rhythmic_pattern="sustained_rise",
            harmonic_content=["V_pedal"]  # Dominant pedal
        )


# ============================================================================
# TURNAROUNDS
# ============================================================================

class TurnaroundGenerator:
    """
    Generate turnarounds (short progressions that return to tonic)

    Common in:
    - Jazz (I-vi-ii-V)
    - Blues (I-IV-I-V)
    - Classical (ii-V-I, IV-V-I)
    """

    @staticmethod
    def jazz_turnaround(key: int, is_major: bool = True) -> List[str]:
        """
        Generate jazz turnaround (I-VI-ii-V or I-vi-ii-V)

        Args:
            key: Tonic MIDI note
            is_major: True for major key

        Returns:
            List of chord symbols
        """
        if is_major:
            return ["Imaj7", "vi7", "ii7", "V7"]
        else:
            return ["i7", "VI7", "ii7b5", "V7"]

    @staticmethod
    def blues_turnaround(key: int) -> List[str]:
        """
        Generate blues turnaround

        Args:
            key: Tonic MIDI note

        Returns:
            List of chord symbols
        """
        return ["I7", "VI7", "ii7", "V7"]

    @staticmethod
    def classical_authentic_cadence(key: int, is_major: bool = True) -> List[str]:
        """
        Generate authentic cadence (V-I or V7-I)

        Args:
            key: Tonic MIDI note
            is_major: True for major

        Returns:
            List of chord symbols
        """
        if is_major:
            return ["V7", "I"]
        else:
            return ["V7", "i"]

    @staticmethod
    def classical_plagal_cadence(key: int, is_major: bool = True) -> List[str]:
        """
        Generate plagal cadence (IV-I, "Amen" cadence)

        Args:
            key: Tonic MIDI note
            is_major: True for major

        Returns:
            List of chord symbols
        """
        if is_major:
            return ["IV", "I"]
        else:
            return ["iv", "i"]

    @staticmethod
    def deceptive_cadence(key: int, is_major: bool = True) -> List[str]:
        """
        Generate deceptive cadence (V-vi instead of V-I)

        Args:
            key: Tonic MIDI note
            is_major: True for major

        Returns:
            List of chord symbols
        """
        if is_major:
            return ["V7", "vi"]
        else:
            return ["V7", "VI"]


# ============================================================================
# MAIN TRANSITION ENGINE CLASS
# ============================================================================

class TransitionEngine:
    """
    Main transition engine - high-level API for generating transitions
    """

    @staticmethod
    def generate_modulation(
        from_key: int,
        from_major: bool,
        to_key: int,
        to_major: bool,
        technique: ModulationType = ModulationType.COMMON_CHORD,
        **kwargs
    ) -> Modulation:
        """
        Generate a modulation between keys

        Args:
            from_key: Starting key (MIDI note)
            from_major: True if starting in major
            to_key: Destination key (MIDI note)
            to_major: True if destination is major
            technique: Modulation technique to use
            **kwargs: Additional arguments for specific technique

        Returns:
            Modulation object
        """
        if technique == ModulationType.COMMON_CHORD:
            return CommonChordModulation.generate(
                from_key, from_major, to_key, to_major, **kwargs
            )
        elif technique == ModulationType.DIRECT:
            return DirectModulation.generate(
                from_key, from_major, to_key, to_major, **kwargs
            )
        elif technique == ModulationType.SEQUENTIAL:
            return SequentialModulation.generate(
                from_key, from_major, to_key, to_major, **kwargs
            )
        elif technique == ModulationType.ENHARMONIC:
            return EnharmonicModulation.generate(
                from_key, from_major, to_key, to_major
            )
        elif technique == ModulationType.CHROMATIC_MEDIANT:
            return ChromaticMediantModulation.generate(
                from_key, from_major, to_key, to_major
            )
        else:
            # Default to common chord
            return CommonChordModulation.generate(
                from_key, from_major, to_key, to_major
            )

    @staticmethod
    def generate_transition(
        transition_type: TransitionType,
        length_bars: int = 4,
        **kwargs
    ) -> Transition:
        """
        Generate a section transition

        Args:
            transition_type: Type of transition
            length_bars: Number of bars
            **kwargs: Additional arguments for specific transition type

        Returns:
            Transition object
        """
        if transition_type == TransitionType.BUILD_UP:
            return BuildUpTransition.generate(length_bars, **kwargs)
        elif transition_type == TransitionType.BREAKDOWN:
            return BreakdownTransition.generate(length_bars, **kwargs)
        elif transition_type == TransitionType.FILL:
            return DrumFillTransition.generate(length_bars, **kwargs)
        elif transition_type == TransitionType.RISER:
            return RiserTransition.generate(length_bars, **kwargs)
        else:
            # Default transition
            return Transition(
                transition_type=transition_type,
                length_bars=length_bars,
                start_dynamic=0.5,
                end_dynamic=0.5
            )

    @staticmethod
    def generate_turnaround(
        key: int,
        is_major: bool = True,
        style: str = "jazz"
    ) -> List[str]:
        """
        Generate a turnaround progression

        Args:
            key: Tonic MIDI note
            is_major: True for major key
            style: Style of turnaround ('jazz', 'blues', 'classical')

        Returns:
            List of chord symbols
        """
        if style == "jazz":
            return TurnaroundGenerator.jazz_turnaround(key, is_major)
        elif style == "blues":
            return TurnaroundGenerator.blues_turnaround(key)
        elif style == "authentic":
            return TurnaroundGenerator.classical_authentic_cadence(key, is_major)
        elif style == "plagal":
            return TurnaroundGenerator.classical_plagal_cadence(key, is_major)
        elif style == "deceptive":
            return TurnaroundGenerator.deceptive_cadence(key, is_major)
        else:
            # Default to jazz
            return TurnaroundGenerator.jazz_turnaround(key, is_major)

    @staticmethod
    def analyze_key_relationship(key1: int, is_major1: bool, key2: int, is_major2: bool) -> str:
        """
        Analyze relationship between two keys

        Args:
            key1: First key tonic
            is_major1: True if first key is major
            key2: Second key tonic
            is_major2: True if second key is major

        Returns:
            String describing relationship
        """
        interval = (key2 - key1) % 12

        # Check for parallel/relative relationships
        if key1 % 12 == key2 % 12:
            if is_major1 != is_major2:
                return "Parallel major/minor"
            else:
                return "Same key"

        # Check for relative major/minor
        if is_major1 and not is_major2 and interval == 9:
            return "Relative minor"
        elif not is_major1 and is_major2 and interval == 3:
            return "Relative major"

        # Check for common relationships
        relationships = {
            0: "Same key",
            1: "Chromatic neighbor",
            2: "Whole step",
            3: "Minor third (chromatic mediant)",
            4: "Major third (chromatic mediant)",
            5: "Subdominant",
            6: "Tritone",
            7: "Dominant",
            8: "Minor sixth",
            9: "Submediant",
            10: "Minor seventh",
            11: "Leading tone"
        }

        return relationships.get(interval, "Distant key")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🎵 TRANSITION ENGINE - Modulation & Transition System\n")

    # Example 1: Common Chord Modulation
    print("=" * 80)
    print("EXAMPLE 1: Common Chord Modulation (C major → G major)")
    print("=" * 80)
    mod1 = TransitionEngine.generate_modulation(
        from_key=60,  # C
        from_major=True,
        to_key=67,  # G
        to_major=True,
        technique=ModulationType.COMMON_CHORD,
        length_bars=4
    )
    print(f"Technique: {mod1.technique.value}")
    print(f"From: C major → To: G major")
    print(f"Pivot chords: {mod1.pivot_chords}")
    print(f"Length: {mod1.length_bars} bars")

    # Example 2: Chromatic Mediant Modulation
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Chromatic Mediant Modulation (C major → E major)")
    print("=" * 80)
    mod2 = TransitionEngine.generate_modulation(
        from_key=60,  # C
        from_major=True,
        to_key=64,  # E
        to_major=True,
        technique=ModulationType.CHROMATIC_MEDIANT
    )
    print(f"Technique: {mod2.technique.value}")
    print(f"From: C major → To: E major (cinematic effect)")
    relationship = TransitionEngine.analyze_key_relationship(60, True, 64, True)
    print(f"Relationship: {relationship}")

    # Example 3: Build-up Transition
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Build-up Transition (Verse → Chorus)")
    print("=" * 80)
    trans1 = TransitionEngine.generate_transition(
        transition_type=TransitionType.BUILD_UP,
        length_bars=4,
        intensity=0.9
    )
    print(f"Type: {trans1.transition_type.value}")
    print(f"Length: {trans1.length_bars} bars")
    print(f"Dynamic: {trans1.start_dynamic:.2f} → {trans1.end_dynamic:.2f}")
    print(f"Texture: {trans1.start_texture:.2f} → {trans1.end_texture:.2f}")
    print(f"Rhythm: {trans1.rhythmic_pattern}")

    # Example 4: Breakdown Transition
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Breakdown Transition (Chorus → Bridge)")
    print("=" * 80)
    trans2 = TransitionEngine.generate_transition(
        transition_type=TransitionType.BREAKDOWN,
        length_bars=4,
        final_texture=0.2
    )
    print(f"Type: {trans2.transition_type.value}")
    print(f"Length: {trans2.length_bars} bars")
    print(f"Dynamic: {trans2.start_dynamic:.2f} → {trans2.end_dynamic:.2f}")
    print(f"Texture: {trans2.start_texture:.2f} → {trans2.end_texture:.2f}")

    # Example 5: Jazz Turnaround
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Jazz Turnaround (I-vi-ii-V)")
    print("=" * 80)
    turnaround_jazz = TransitionEngine.generate_turnaround(
        key=60,  # C
        is_major=True,
        style="jazz"
    )
    print(f"Jazz turnaround in C major: {' → '.join(turnaround_jazz)}")

    # Example 6: Blues Turnaround
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Blues Turnaround")
    print("=" * 80)
    turnaround_blues = TransitionEngine.generate_turnaround(
        key=64,  # E
        is_major=False,
        style="blues"
    )
    print(f"Blues turnaround in E: {' → '.join(turnaround_blues)}")

    # Example 7: EDM Riser
    print("\n" + "=" * 80)
    print("EXAMPLE 7: EDM Riser Transition")
    print("=" * 80)
    riser = TransitionEngine.generate_transition(
        transition_type=TransitionType.RISER,
        length_bars=8,
        start_pitch=48,
        end_pitch=72
    )
    print(f"Type: {riser.transition_type.value}")
    print(f"Length: {riser.length_bars} bars")
    print(f"Dynamic: {riser.start_dynamic:.2f} → {riser.end_dynamic:.2f}")
    print(f"Creates tension for drop/chorus")

    # Example 8: Key Relationship Analysis
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Key Relationship Analysis")
    print("=" * 80)
    relationships = [
        (60, True, 67, True, "C major → G major"),
        (60, True, 60, False, "C major → C minor"),
        (60, True, 69, False, "C major → A minor"),
        (60, True, 64, True, "C major → E major"),
        (60, True, 68, True, "C major → Ab major"),
    ]

    for key1, maj1, key2, maj2, description in relationships:
        rel = TransitionEngine.analyze_key_relationship(key1, maj1, key2, maj2)
        print(f"{description}: {rel}")

    print("\n✅ Transition Engine examples complete!")
    print("This module provides smooth transitions and modulations between sections.\n")
