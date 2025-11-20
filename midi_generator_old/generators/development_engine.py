#!/usr/bin/env python3
"""
Development Engine - Motivic Development & Thematic Transformation
Part of the Ultimate MIDI Generation Library

This module provides comprehensive motivic development techniques:
- Repetition (exact, varied)
- Transposition (sequence)
- Inversion (melodic mirror)
- Retrograde (reverse)
- Augmentation (slower)
- Diminution (faster)
- Fragmentation (use part of motif)
- Extension (add notes)
- Combination (multiple techniques)
- Thematic transformation (Liszt-style metamorphosis)

Author: Agent 5 - Form & Structure Engine
Research: Schoenberg's Fundamentals, Reti's Thematic Process in Music, Caplin's Classical Form
"""

import copy
import random
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class DevelopmentTechnique(Enum):
    """Types of motivic development techniques"""
    REPETITION = "repetition"
    TRANSPOSITION = "transposition"
    SEQUENCE = "sequence"
    INVERSION = "inversion"
    RETROGRADE = "retrograde"
    RETROGRADE_INVERSION = "retrograde_inversion"
    AUGMENTATION = "augmentation"
    DIMINUTION = "diminution"
    FRAGMENTATION = "fragmentation"
    EXTENSION = "extension"
    INTERPOLATION = "interpolation"
    RHYTHMIC_SHIFT = "rhythmic_shift"
    INTERVALLIC_EXPANSION = "intervallic_expansion"
    INTERVALLIC_CONTRACTION = "intervallic_contraction"
    OCTAVE_DISPLACEMENT = "octave_displacement"


@dataclass
class Motif:
    """
    Represents a musical motif (short melodic/rhythmic idea)

    Attributes:
        pitches: List of MIDI note numbers
        durations: List of note durations (in beats)
        name: Identifier for this motif
        character: Musical character/mood
    """
    pitches: List[int]
    durations: List[float]
    name: str = "motif"
    character: str = "neutral"

    def __post_init__(self):
        if len(self.pitches) != len(self.durations):
            raise ValueError("Pitches and durations must have same length")

    @property
    def length(self) -> int:
        """Number of notes in motif"""
        return len(self.pitches)

    @property
    def total_duration(self) -> float:
        """Total duration in beats"""
        return sum(self.durations)

    @property
    def intervals(self) -> List[int]:
        """Intervals between consecutive notes (in semitones)"""
        if len(self.pitches) < 2:
            return []
        return [self.pitches[i+1] - self.pitches[i] for i in range(len(self.pitches)-1)]

    def copy(self) -> 'Motif':
        """Create a deep copy of this motif"""
        return Motif(
            pitches=self.pitches.copy(),
            durations=self.durations.copy(),
            name=self.name,
            character=self.character
        )


@dataclass
class DevelopedMotif:
    """
    A motif that has been developed/transformed

    Attributes:
        original_motif: Reference to original
        developed_motif: The transformed motif
        technique: Technique(s) applied
        parameters: Parameters used for development
    """
    original_motif: Motif
    developed_motif: Motif
    technique: DevelopmentTechnique
    parameters: Dict = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


# ============================================================================
# BASIC DEVELOPMENT TECHNIQUES
# ============================================================================

class MotifTransformations:
    """Basic transformations on motifs"""

    @staticmethod
    def transpose(motif: Motif, semitones: int) -> Motif:
        """
        Transpose motif by semitones

        Args:
            motif: Original motif
            semitones: Number of semitones to transpose (+/-)

        Returns:
            Transposed motif
        """
        new_motif = motif.copy()
        new_motif.pitches = [p + semitones for p in motif.pitches]
        new_motif.name = f"{motif.name}_T{semitones:+d}"
        return new_motif

    @staticmethod
    def invert(motif: Motif, axis: Optional[int] = None) -> Motif:
        """
        Invert motif around an axis pitch

        Args:
            motif: Original motif
            axis: Axis pitch for inversion (default: first note)

        Returns:
            Inverted motif
        """
        if axis is None:
            axis = motif.pitches[0]

        new_motif = motif.copy()
        new_motif.pitches = [axis - (p - axis) for p in motif.pitches]
        new_motif.name = f"{motif.name}_inv"
        new_motif.character = f"inverted {motif.character}"
        return new_motif

    @staticmethod
    def retrograde(motif: Motif) -> Motif:
        """
        Reverse motif (play backwards)

        Args:
            motif: Original motif

        Returns:
            Reversed motif
        """
        new_motif = motif.copy()
        new_motif.pitches = list(reversed(motif.pitches))
        new_motif.durations = list(reversed(motif.durations))
        new_motif.name = f"{motif.name}_retro"
        return new_motif

    @staticmethod
    def retrograde_inversion(motif: Motif, axis: Optional[int] = None) -> Motif:
        """
        Invert and then reverse motif

        Args:
            motif: Original motif
            axis: Axis for inversion

        Returns:
            Retrograde-inverted motif
        """
        inverted = MotifTransformations.invert(motif, axis)
        return MotifTransformations.retrograde(inverted)

    @staticmethod
    def augment(motif: Motif, factor: float = 2.0) -> Motif:
        """
        Augment motif (make durations longer)

        Args:
            motif: Original motif
            factor: Multiplication factor for durations

        Returns:
            Augmented motif
        """
        new_motif = motif.copy()
        new_motif.durations = [d * factor for d in motif.durations]
        new_motif.name = f"{motif.name}_aug{factor}"
        new_motif.character = f"stately {motif.character}"
        return new_motif

    @staticmethod
    def diminish(motif: Motif, factor: float = 0.5) -> Motif:
        """
        Diminish motif (make durations shorter)

        Args:
            motif: Original motif
            factor: Multiplication factor for durations

        Returns:
            Diminished motif
        """
        new_motif = motif.copy()
        new_motif.durations = [d * factor for d in motif.durations]
        new_motif.name = f"{motif.name}_dim{factor}"
        new_motif.character = f"urgent {motif.character}"
        return new_motif

    @staticmethod
    def fragment(motif: Motif, start_note: int = 0, num_notes: Optional[int] = None) -> Motif:
        """
        Extract fragment of motif

        Args:
            motif: Original motif
            start_note: Starting note index
            num_notes: Number of notes to extract (None = to end)

        Returns:
            Fragmented motif
        """
        if num_notes is None:
            num_notes = motif.length - start_note

        end_note = min(start_note + num_notes, motif.length)

        new_motif = motif.copy()
        new_motif.pitches = motif.pitches[start_note:end_note]
        new_motif.durations = motif.durations[start_note:end_note]
        new_motif.name = f"{motif.name}_frag{start_note}-{end_note}"
        return new_motif

    @staticmethod
    def extend(motif: Motif, extension_pitches: List[int], extension_durations: List[float]) -> Motif:
        """
        Extend motif by adding notes

        Args:
            motif: Original motif
            extension_pitches: Pitches to add
            extension_durations: Durations to add

        Returns:
            Extended motif
        """
        new_motif = motif.copy()
        new_motif.pitches.extend(extension_pitches)
        new_motif.durations.extend(extension_durations)
        new_motif.name = f"{motif.name}_ext"
        return new_motif

    @staticmethod
    def rhythmic_shift(motif: Motif, shift_beats: float) -> Motif:
        """
        Shift motif rhythmically (change onset time)

        Args:
            motif: Original motif
            shift_beats: Amount to shift (in beats)

        Returns:
            Rhythmically shifted motif (note: onset change must be handled by caller)
        """
        # This primarily affects onset time when placed in context
        new_motif = motif.copy()
        new_motif.name = f"{motif.name}_shift{shift_beats:+.2f}"
        return new_motif


# ============================================================================
# ADVANCED DEVELOPMENT TECHNIQUES
# ============================================================================

class AdvancedDevelopment:
    """More sophisticated development techniques"""

    @staticmethod
    def sequence(motif: Motif, num_repetitions: int = 3, interval: int = 2,
                 ascending: bool = True) -> List[Motif]:
        """
        Create sequence (transposed repetitions)

        Args:
            motif: Original motif
            num_repetitions: Number of times to repeat
            interval: Interval for transposition (semitones)
            ascending: True for ascending, False for descending

        Returns:
            List of transposed motifs
        """
        sequence_motifs = [motif]
        current_transposition = 0

        for i in range(num_repetitions - 1):
            if ascending:
                current_transposition += interval
            else:
                current_transposition -= interval

            transposed = MotifTransformations.transpose(motif, current_transposition)
            transposed.name = f"{motif.name}_seq{i+1}"
            sequence_motifs.append(transposed)

        return sequence_motifs

    @staticmethod
    def intervallic_expansion(motif: Motif, expansion_factor: float = 1.5) -> Motif:
        """
        Expand intervals between notes

        Args:
            motif: Original motif
            expansion_factor: Factor to multiply intervals by

        Returns:
            Motif with expanded intervals
        """
        if len(motif.pitches) < 2:
            return motif.copy()

        new_pitches = [motif.pitches[0]]  # Keep first note
        for i in range(1, len(motif.pitches)):
            interval = motif.pitches[i] - motif.pitches[i-1]
            expanded_interval = int(interval * expansion_factor)
            new_pitches.append(new_pitches[-1] + expanded_interval)

        new_motif = motif.copy()
        new_motif.pitches = new_pitches
        new_motif.name = f"{motif.name}_exp{expansion_factor}"
        return new_motif

    @staticmethod
    def intervallic_contraction(motif: Motif, contraction_factor: float = 0.5) -> Motif:
        """
        Contract intervals between notes

        Args:
            motif: Original motif
            contraction_factor: Factor to multiply intervals by

        Returns:
            Motif with contracted intervals
        """
        return AdvancedDevelopment.intervallic_expansion(motif, contraction_factor)

    @staticmethod
    def octave_displacement(motif: Motif, displacement_pattern: Optional[List[int]] = None) -> Motif:
        """
        Displace notes to different octaves

        Args:
            motif: Original motif
            displacement_pattern: List of octave displacements (+/- octaves)
                                If None, random displacement

        Returns:
            Motif with octave displacement
        """
        new_motif = motif.copy()

        if displacement_pattern is None:
            # Random displacement
            displacement_pattern = [random.choice([-1, 0, 1]) for _ in range(motif.length)]

        # Ensure pattern matches length
        while len(displacement_pattern) < motif.length:
            displacement_pattern.append(0)

        new_motif.pitches = [
            p + (disp * 12)
            for p, disp in zip(motif.pitches, displacement_pattern[:motif.length])
        ]
        new_motif.name = f"{motif.name}_octdisp"
        return new_motif

    @staticmethod
    def interpolation(motif1: Motif, motif2: Motif, num_steps: int = 3) -> List[Motif]:
        """
        Interpolate between two motifs

        Args:
            motif1: Starting motif
            motif2: Ending motif
            num_steps: Number of intermediate steps

        Returns:
            List of interpolated motifs
        """
        # Ensure motifs have same length (truncate or extend as needed)
        max_len = max(motif1.length, motif2.length)

        # Extend shorter motif by repeating last note
        pitches1 = motif1.pitches + [motif1.pitches[-1]] * (max_len - motif1.length)
        pitches2 = motif2.pitches + [motif2.pitches[-1]] * (max_len - motif2.length)

        durations1 = motif1.durations + [motif1.durations[-1]] * (max_len - motif1.length)
        durations2 = motif2.durations + [motif2.durations[-1]] * (max_len - motif2.length)

        interpolated = []

        for step in range(num_steps + 2):
            alpha = step / (num_steps + 1)  # 0 to 1

            # Interpolate pitches
            interp_pitches = [
                int(p1 * (1 - alpha) + p2 * alpha)
                for p1, p2 in zip(pitches1, pitches2)
            ]

            # Interpolate durations
            interp_durations = [
                d1 * (1 - alpha) + d2 * alpha
                for d1, d2 in zip(durations1, durations2)
            ]

            new_motif = Motif(
                pitches=interp_pitches,
                durations=interp_durations,
                name=f"interp_{step}",
                character=f"blend of {motif1.character} and {motif2.character}"
            )
            interpolated.append(new_motif)

        return interpolated


# ============================================================================
# THEMATIC TRANSFORMATION (Liszt-style)
# ============================================================================

class ThematicTransformation:
    """
    Liszt-style thematic transformation (metamorphosis)

    Transform a theme's character while maintaining recognizability
    """

    @staticmethod
    def heroic_transformation(motif: Motif) -> Motif:
        """
        Transform theme to heroic character

        Techniques:
        - Augmentation (slower, grander)
        - Lower register
        - Wider intervals
        """
        transformed = MotifTransformations.augment(motif, 1.5)
        transformed = AdvancedDevelopment.intervallic_expansion(transformed, 1.3)
        transformed.pitches = [p - 12 for p in transformed.pitches]  # Lower octave
        transformed.character = "heroic, triumphant"
        transformed.name = f"{motif.name}_heroic"
        return transformed

    @staticmethod
    def lyrical_transformation(motif: Motif) -> Motif:
        """
        Transform theme to lyrical character

        Techniques:
        - Smoother intervals
        - Moderate register
        - Flowing rhythm
        """
        transformed = AdvancedDevelopment.intervallic_contraction(motif, 0.7)
        transformed.character = "lyrical, singing"
        transformed.name = f"{motif.name}_lyrical"
        return transformed

    @staticmethod
    def dramatic_transformation(motif: Motif) -> Motif:
        """
        Transform theme to dramatic character

        Techniques:
        - Wider leaps
        - Rhythmic variation
        - Extremes of register
        """
        transformed = AdvancedDevelopment.intervallic_expansion(motif, 1.8)
        transformed = AdvancedDevelopment.octave_displacement(transformed)
        transformed.character = "dramatic, intense"
        transformed.name = f"{motif.name}_dramatic"
        return transformed

    @staticmethod
    def pastoral_transformation(motif: Motif) -> Motif:
        """
        Transform theme to pastoral character

        Techniques:
        - Gentle contours
        - Middle register
        - Relaxed rhythm
        """
        transformed = AdvancedDevelopment.intervallic_contraction(motif, 0.8)
        transformed = MotifTransformations.augment(transformed, 1.2)
        transformed.character = "pastoral, peaceful"
        transformed.name = f"{motif.name}_pastoral"
        return transformed

    @staticmethod
    def march_transformation(motif: Motif) -> Motif:
        """
        Transform theme to march character

        Techniques:
        - Emphatic rhythm
        - Repetition
        - Strong downbeats
        """
        # Normalize durations to march-like pattern
        transformed = motif.copy()
        # Make all durations equal (march-like)
        avg_duration = sum(motif.durations) / len(motif.durations)
        transformed.durations = [avg_duration] * len(motif.durations)
        transformed.character = "march, militaristic"
        transformed.name = f"{motif.name}_march"
        return transformed


# ============================================================================
# DEVELOPMENT CHAINS & COMBINATIONS
# ============================================================================

class DevelopmentChain:
    """
    Chain multiple development techniques together
    """

    @staticmethod
    def apply_chain(motif: Motif, techniques: List[Tuple[str, Dict]]) -> Motif:
        """
        Apply chain of development techniques

        Args:
            motif: Original motif
            techniques: List of (technique_name, parameters) tuples

        Returns:
            Fully developed motif

        Example:
            techniques = [
                ('transpose', {'semitones': 5}),
                ('invert', {}),
                ('augment', {'factor': 1.5})
            ]
        """
        current_motif = motif

        for technique_name, params in techniques:
            if technique_name == 'transpose':
                current_motif = MotifTransformations.transpose(current_motif, **params)
            elif technique_name == 'invert':
                current_motif = MotifTransformations.invert(current_motif, **params)
            elif technique_name == 'retrograde':
                current_motif = MotifTransformations.retrograde(current_motif)
            elif technique_name == 'augment':
                current_motif = MotifTransformations.augment(current_motif, **params)
            elif technique_name == 'diminish':
                current_motif = MotifTransformations.diminish(current_motif, **params)
            elif technique_name == 'fragment':
                current_motif = MotifTransformations.fragment(current_motif, **params)
            elif technique_name == 'intervallic_expansion':
                current_motif = AdvancedDevelopment.intervallic_expansion(current_motif, **params)
            elif technique_name == 'intervallic_contraction':
                current_motif = AdvancedDevelopment.intervallic_contraction(current_motif, **params)
            elif technique_name == 'octave_displacement':
                current_motif = AdvancedDevelopment.octave_displacement(current_motif, **params)

        return current_motif


# ============================================================================
# MAIN DEVELOPMENT ENGINE CLASS
# ============================================================================

class DevelopmentEngine:
    """
    Main development engine - high-level API for motivic development
    """

    @staticmethod
    def develop_motif(
        motif: Motif,
        technique: DevelopmentTechnique,
        **kwargs
    ) -> DevelopedMotif:
        """
        Develop a motif using specified technique

        Args:
            motif: Original motif
            technique: Development technique to apply
            **kwargs: Parameters for specific technique

        Returns:
            DevelopedMotif object
        """
        if technique == DevelopmentTechnique.REPETITION:
            developed = motif.copy()

        elif technique == DevelopmentTechnique.TRANSPOSITION:
            semitones = kwargs.get('semitones', 5)
            developed = MotifTransformations.transpose(motif, semitones)

        elif technique == DevelopmentTechnique.INVERSION:
            axis = kwargs.get('axis', None)
            developed = MotifTransformations.invert(motif, axis)

        elif technique == DevelopmentTechnique.RETROGRADE:
            developed = MotifTransformations.retrograde(motif)

        elif technique == DevelopmentTechnique.RETROGRADE_INVERSION:
            axis = kwargs.get('axis', None)
            developed = MotifTransformations.retrograde_inversion(motif, axis)

        elif technique == DevelopmentTechnique.AUGMENTATION:
            factor = kwargs.get('factor', 2.0)
            developed = MotifTransformations.augment(motif, factor)

        elif technique == DevelopmentTechnique.DIMINUTION:
            factor = kwargs.get('factor', 0.5)
            developed = MotifTransformations.diminish(motif, factor)

        elif technique == DevelopmentTechnique.FRAGMENTATION:
            start_note = kwargs.get('start_note', 0)
            num_notes = kwargs.get('num_notes', None)
            developed = MotifTransformations.fragment(motif, start_note, num_notes)

        elif technique == DevelopmentTechnique.INTERVALLIC_EXPANSION:
            factor = kwargs.get('expansion_factor', 1.5)
            developed = AdvancedDevelopment.intervallic_expansion(motif, factor)

        elif technique == DevelopmentTechnique.INTERVALLIC_CONTRACTION:
            factor = kwargs.get('contraction_factor', 0.5)
            developed = AdvancedDevelopment.intervallic_contraction(motif, factor)

        elif technique == DevelopmentTechnique.OCTAVE_DISPLACEMENT:
            pattern = kwargs.get('displacement_pattern', None)
            developed = AdvancedDevelopment.octave_displacement(motif, pattern)

        else:
            developed = motif.copy()

        return DevelopedMotif(
            original_motif=motif,
            developed_motif=developed,
            technique=technique,
            parameters=kwargs
        )

    @staticmethod
    def create_development_section(
        motif: Motif,
        num_variations: int = 8,
        techniques: Optional[List[DevelopmentTechnique]] = None
    ) -> List[DevelopedMotif]:
        """
        Create a complete development section with multiple variations

        Args:
            motif: Original motif
            num_variations: Number of variations to generate
            techniques: List of techniques to use (random if None)

        Returns:
            List of developed motifs
        """
        if techniques is None:
            # Default set of techniques for development
            techniques = [
                DevelopmentTechnique.TRANSPOSITION,
                DevelopmentTechnique.INVERSION,
                DevelopmentTechnique.FRAGMENTATION,
                DevelopmentTechnique.AUGMENTATION,
                DevelopmentTechnique.SEQUENCE,
                DevelopmentTechnique.INTERVALLIC_EXPANSION,
            ]

        developed_motifs = []

        for i in range(num_variations):
            # Select technique (cycle through available)
            technique = techniques[i % len(techniques)]

            # Apply technique with varying parameters
            if technique == DevelopmentTechnique.TRANSPOSITION:
                semitones = random.choice([2, 3, 5, 7, -2, -3, -5])
                developed = DevelopmentEngine.develop_motif(motif, technique, semitones=semitones)

            elif technique == DevelopmentTechnique.FRAGMENTATION:
                max_start = max(0, motif.length - 2)
                start_note = random.randint(0, max_start)
                num_notes = random.randint(2, motif.length - start_note)
                developed = DevelopmentEngine.develop_motif(
                    motif, technique, start_note=start_note, num_notes=num_notes
                )

            elif technique == DevelopmentTechnique.AUGMENTATION:
                factor = random.choice([1.5, 2.0, 2.5])
                developed = DevelopmentEngine.develop_motif(motif, technique, factor=factor)

            else:
                developed = DevelopmentEngine.develop_motif(motif, technique)

            developed_motifs.append(developed)

        return developed_motifs

    @staticmethod
    def analyze_motif(motif: Motif) -> Dict[str, any]:
        """
        Analyze characteristics of a motif

        Args:
            motif: Motif to analyze

        Returns:
            Dictionary of analysis results
        """
        intervals = motif.intervals

        analysis = {
            'length': motif.length,
            'total_duration': motif.total_duration,
            'pitch_range': max(motif.pitches) - min(motif.pitches) if motif.pitches else 0,
            'lowest_pitch': min(motif.pitches) if motif.pitches else None,
            'highest_pitch': max(motif.pitches) if motif.pitches else None,
            'average_pitch': sum(motif.pitches) / len(motif.pitches) if motif.pitches else 0,
            'intervals': intervals,
            'interval_variety': len(set(abs(i) for i in intervals)) if intervals else 0,
            'predominantly_stepwise': sum(1 for i in intervals if abs(i) <= 2) / len(intervals) if intervals else 0,
            'has_leaps': any(abs(i) > 4 for i in intervals) if intervals else False,
            'contour': DevelopmentEngine._analyze_contour(motif.pitches),
            'rhythmic_variety': len(set(motif.durations)) / len(motif.durations) if motif.durations else 0,
        }

        return analysis

    @staticmethod
    def _analyze_contour(pitches: List[int]) -> str:
        """Analyze melodic contour"""
        if len(pitches) < 3:
            return "too short"

        # Calculate general direction
        start = pitches[0]
        middle = pitches[len(pitches)//2]
        end = pitches[-1]

        if middle > start and middle > end:
            return "arch"
        elif middle < start and middle < end:
            return "valley"
        elif end > start:
            return "ascending"
        elif end < start:
            return "descending"
        else:
            return "static"


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("\n🎵 DEVELOPMENT ENGINE - Motivic Development & Transformation\n")

    # Create a simple motif
    original_motif = Motif(
        pitches=[60, 62, 64, 65, 67],  # C D E F G
        durations=[1.0, 1.0, 1.0, 1.0, 2.0],
        name="original_theme",
        character="simple, ascending"
    )

    print("=" * 80)
    print("ORIGINAL MOTIF")
    print("=" * 80)
    print(f"Name: {original_motif.name}")
    print(f"Pitches: {original_motif.pitches}")
    print(f"Durations: {original_motif.durations}")
    print(f"Character: {original_motif.character}")
    analysis = DevelopmentEngine.analyze_motif(original_motif)
    print(f"\nAnalysis:")
    print(f"  Length: {analysis['length']} notes")
    print(f"  Range: {analysis['pitch_range']} semitones")
    print(f"  Contour: {analysis['contour']}")
    print(f"  Intervals: {original_motif.intervals}")

    # Example 1: Transposition
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Transposition (up 7 semitones)")
    print("=" * 80)
    transposed = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.TRANSPOSITION,
        semitones=7
    )
    print(f"Original: {original_motif.pitches}")
    print(f"Transposed: {transposed.developed_motif.pitches}")

    # Example 2: Inversion
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Melodic Inversion")
    print("=" * 80)
    inverted = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.INVERSION
    )
    print(f"Original: {original_motif.pitches}")
    print(f"Inverted: {inverted.developed_motif.pitches}")

    # Example 3: Retrograde
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Retrograde (reverse)")
    print("=" * 80)
    retrograde = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.RETROGRADE
    )
    print(f"Original: {original_motif.pitches}")
    print(f"Retrograde: {retrograde.developed_motif.pitches}")

    # Example 4: Augmentation
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Augmentation (2x slower)")
    print("=" * 80)
    augmented = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.AUGMENTATION,
        factor=2.0
    )
    print(f"Original durations: {original_motif.durations}")
    print(f"Augmented durations: {augmented.developed_motif.durations}")

    # Example 5: Diminution
    print("\n" + "=" * 80)
    print("EXAMPLE 5: Diminution (2x faster)")
    print("=" * 80)
    diminished = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.DIMINUTION,
        factor=0.5
    )
    print(f"Original durations: {original_motif.durations}")
    print(f"Diminished durations: {diminished.developed_motif.durations}")

    # Example 6: Fragmentation
    print("\n" + "=" * 80)
    print("EXAMPLE 6: Fragmentation (first 3 notes)")
    print("=" * 80)
    fragmented = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.FRAGMENTATION,
        start_note=0,
        num_notes=3
    )
    print(f"Original: {original_motif.pitches}")
    print(f"Fragment: {fragmented.developed_motif.pitches}")

    # Example 7: Sequence
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Sequence (ascending by major 2nds)")
    print("=" * 80)
    sequence_motifs = AdvancedDevelopment.sequence(
        original_motif,
        num_repetitions=4,
        interval=2,
        ascending=True
    )
    print("Sequence:")
    for i, seq_motif in enumerate(sequence_motifs):
        print(f"  Step {i}: {seq_motif.pitches}")

    # Example 8: Intervallic Expansion
    print("\n" + "=" * 80)
    print("EXAMPLE 8: Intervallic Expansion (1.5x wider)")
    print("=" * 80)
    expanded = DevelopmentEngine.develop_motif(
        original_motif,
        DevelopmentTechnique.INTERVALLIC_EXPANSION,
        expansion_factor=1.5
    )
    print(f"Original: {original_motif.pitches} (intervals: {original_motif.intervals})")
    print(f"Expanded: {expanded.developed_motif.pitches} (intervals: {expanded.developed_motif.intervals})")

    # Example 9: Thematic Transformation
    print("\n" + "=" * 80)
    print("EXAMPLE 9: Thematic Transformations (Liszt-style)")
    print("=" * 80)

    heroic = ThematicTransformation.heroic_transformation(original_motif)
    print(f"Heroic: {heroic.pitches} - {heroic.character}")

    lyrical = ThematicTransformation.lyrical_transformation(original_motif)
    print(f"Lyrical: {lyrical.pitches} - {lyrical.character}")

    dramatic = ThematicTransformation.dramatic_transformation(original_motif)
    print(f"Dramatic: {dramatic.pitches} - {dramatic.character}")

    # Example 10: Development Chain
    print("\n" + "=" * 80)
    print("EXAMPLE 10: Development Chain (transpose → invert → augment)")
    print("=" * 80)
    chain_techniques = [
        ('transpose', {'semitones': 7}),
        ('invert', {}),
        ('augment', {'factor': 1.5})
    ]
    chained = DevelopmentChain.apply_chain(original_motif, chain_techniques)
    print(f"Original: {original_motif.pitches} | durations: {original_motif.durations}")
    print(f"After chain: {chained.pitches} | durations: {chained.durations}")

    # Example 11: Complete Development Section
    print("\n" + "=" * 80)
    print("EXAMPLE 11: Complete Development Section (8 variations)")
    print("=" * 80)
    development_section = DevelopmentEngine.create_development_section(
        original_motif,
        num_variations=8
    )
    print("Development section with 8 variations:")
    for i, dev_motif in enumerate(development_section):
        print(f"  Variation {i+1} ({dev_motif.technique.value}): {dev_motif.developed_motif.pitches}")

    print("\n✅ Development Engine examples complete!")
    print("This module provides comprehensive motivic development techniques")
    print("for sophisticated compositional development sections.\n")
