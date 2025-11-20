#!/usr/bin/env python3
"""
Universal Voice Leading Optimizer
===================================

A comprehensive voice leading optimization engine that minimizes voice movement
between chords while respecting range constraints. Usable across sax, brass,
strings, vocals, and any multi-voice harmony.

This module implements:
1. Dynamic programming for optimal voicing sequences
2. Multiple voicing types (close, drop-2, drop-3, drop-2-4, spread)
3. Common tone retention algorithms
4. Range-constrained voicing generation
5. Weighted distance metrics (emphasize outer voices)

Theory Background:
------------------
Voice leading principles from classical and jazz theory:
- Minimize total voice movement (parsimonious voice leading)
- Retain common tones in the same voice
- Prefer contrary motion in outer voices
- Avoid large leaps (>octave) unless necessary
- Smooth, singable voice lines

Research References:
--------------------
- Matthew Keating (2023): "An Algorithmic Approach to Jazz Guitar Voice-Leading Chord Fingerings"
  Uses LSTM encoder-decoder for sequence-to-sequence generation with distance minimization
- Mark Levine: "The Jazz Theory Book" - Voice leading chapter
- Dmitri Tymoczko: "A Geometry of Music" - Voice leading spaces and efficient voice leading
- Classical voice leading rules (Fux, Palestrina, Bach)

Applications:
-------------
- Big band sax soli (5 voices, drop-2 preferred)
- Brass section voicing (4-8 voices, various spacings)
- String section (4+ voices, smooth classical voice leading)
- SATB choir (4 voices, vocal range constraints)
- Jazz piano comping (3-5 voices, rootless voicings)

Author: Agent 11 - Voice Leading Optimization Engine
Date: 2025
License: MIT
"""

import math
from typing import List, Tuple, Dict, Optional, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import itertools
from collections import defaultdict

# Note: This module is designed to be standalone and doesn't require
# imports from other modules. It works with simple data structures:
# - Chords are dicts with 'root' and 'quality' keys
# - Voicings are lists of MIDI note numbers


# ============================================================================
# ENUMS AND DATA STRUCTURES
# ============================================================================

class VoicingType(Enum):
    """Voicing types for harmony"""
    CLOSE = "close"                  # All voices within octave (or close spacing)
    DROP_2 = "drop_2"                # Drop 2nd voice from top down octave
    DROP_3 = "drop_3"                # Drop 3rd voice from top down octave
    DROP_2_4 = "drop_2_4"            # Drop 2nd and 4th voices down octave
    SPREAD = "spread"                # Wide spacing throughout
    OPEN = "open"                    # Open position (wider than close)


class MinimizationStrategy(Enum):
    """Strategy for voice leading optimization"""
    TOTAL_MOTION = "total_motion"        # Minimize sum of all voice movements
    MAX_LEAP = "max_leap"                # Minimize largest single voice leap
    WEIGHTED = "weighted"                # Weighted distance (outer voices emphasized)
    COMMON_TONE = "common_tone"          # Maximize common tone retention


@dataclass
class VoiceRange:
    """Range constraints for a single voice"""
    low: int                         # Lowest MIDI note
    high: int                        # Highest MIDI note
    comfortable_low: int             # Comfortable low (prefer this range)
    comfortable_high: int            # Comfortable high

    def __post_init__(self):
        """Validate ranges"""
        if self.low > self.high:
            raise ValueError(f"Invalid range: low ({self.low}) > high ({self.high})")
        if self.comfortable_low < self.low:
            self.comfortable_low = self.low
        if self.comfortable_high > self.high:
            self.comfortable_high = self.high


@dataclass
class Voicing:
    """A specific voicing (list of MIDI pitches)"""
    pitches: List[int]               # MIDI note numbers for each voice
    voicing_type: VoicingType = VoicingType.CLOSE
    cost: float = 0.0                # Cost metric (for DP)

    def __hash__(self):
        """Make hashable for use in sets/dicts"""
        return hash(tuple(self.pitches))

    def __eq__(self, other):
        """Equality comparison"""
        if not isinstance(other, Voicing):
            return False
        return self.pitches == other.pitches


@dataclass
class OptimizationResult:
    """Result from voice leading optimization"""
    voicings: List[Voicing]          # Optimal sequence of voicings
    total_motion: float              # Total semitone movement
    avg_motion: float                # Average motion per chord change
    max_leap: int                    # Largest single voice leap
    motion_per_step: List[float]     # Motion for each chord transition
    common_tones_retained: int       # Number of common tones retained


# ============================================================================
# CHORD ANALYSIS UTILITIES
# ============================================================================

class ChordAnalyzer:
    """Analyze chord structure and extract chord tones"""

    # Chord quality to intervals mapping (semitones from root)
    CHORD_INTERVALS = {
        # Triads
        'major': [0, 4, 7],
        'minor': [0, 3, 7],
        'diminished': [0, 3, 6],
        'augmented': [0, 4, 8],

        # Seventh chords
        'maj7': [0, 4, 7, 11],
        'min7': [0, 3, 7, 10],
        'dom7': [0, 4, 7, 10],
        '7': [0, 4, 7, 10],  # Dominant 7
        'min7b5': [0, 3, 6, 10],  # Half-diminished
        'dim7': [0, 3, 6, 9],
        'aug7': [0, 4, 8, 10],
        'minmaj7': [0, 3, 7, 11],

        # Extensions (9th)
        'maj9': [0, 4, 7, 11, 14],
        'min9': [0, 3, 7, 10, 14],
        'dom9': [0, 4, 7, 10, 14],
        '9': [0, 4, 7, 10, 14],

        # Extensions (11th, 13th)
        'maj13': [0, 4, 7, 11, 14, 21],
        'min11': [0, 3, 7, 10, 14, 17],
        'dom13': [0, 4, 7, 10, 14, 21],
        '13': [0, 4, 7, 10, 14, 21],

        # Suspended
        'sus2': [0, 2, 7],
        'sus4': [0, 5, 7],
        '7sus4': [0, 5, 7, 10],
    }

    @staticmethod
    def get_chord_tones(root: int, quality: str, num_voices: int = 4) -> List[int]:
        """
        Get chord tones (pitch classes 0-11) for a chord.

        Args:
            root: Root pitch class (0-11)
            quality: Chord quality string
            num_voices: Number of voices to generate

        Returns:
            List of pitch classes
        """
        # Normalize quality string
        quality = quality.lower().replace(' ', '').replace('-', '')

        # Get intervals from lookup table
        if quality in ChordAnalyzer.CHORD_INTERVALS:
            intervals = ChordAnalyzer.CHORD_INTERVALS[quality]
        else:
            # Default to major triad if quality unknown
            intervals = [0, 4, 7]

        # Convert to pitch classes
        chord_tones = [(root + interval) % 12 for interval in intervals]

        # If we need more voices than available chord tones, double some tones
        while len(chord_tones) < num_voices:
            # Double root, fifth, or third (in that order of preference)
            if 0 not in chord_tones[len(intervals):]:
                chord_tones.append(root % 12)
            elif 7 in [(root + i) % 12 for i in intervals] and (root + 7) % 12 not in chord_tones[len(intervals):]:
                chord_tones.append((root + 7) % 12)
            else:
                chord_tones.append(chord_tones[1])  # Double third

        return chord_tones[:num_voices]

    @staticmethod
    def get_common_tones(chord1_tones: List[int], chord2_tones: List[int]) -> Set[int]:
        """
        Find common pitch classes between two chords.

        Args:
            chord1_tones: Pitch classes for first chord
            chord2_tones: Pitch classes for second chord

        Returns:
            Set of common pitch classes
        """
        return set(chord1_tones) & set(chord2_tones)


# ============================================================================
# VOICING GENERATOR
# ============================================================================

class VoicingGenerator:
    """Generate all possible voicings for a chord within constraints"""

    @staticmethod
    def generate_all_voicings(
        root: int,
        quality: str,
        num_voices: int,
        voice_ranges: List[VoiceRange],
        voicing_type: VoicingType = VoicingType.CLOSE,
        max_voicings: int = 1000
    ) -> List[Voicing]:
        """
        Generate all valid voicings for a chord.

        Args:
            root: Root pitch class (0-11)
            quality: Chord quality
            num_voices: Number of voices
            voice_ranges: Range constraints for each voice
            voicing_type: Type of voicing to generate
            max_voicings: Maximum number of voicings to generate (performance limit)

        Returns:
            List of valid Voicing objects
        """
        # Get chord tones (pitch classes)
        chord_tones_pc = ChordAnalyzer.get_chord_tones(root, quality, num_voices)

        # Generate based on voicing type
        if voicing_type == VoicingType.CLOSE:
            return VoicingGenerator._generate_close_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )
        elif voicing_type == VoicingType.DROP_2:
            return VoicingGenerator._generate_drop_2_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )
        elif voicing_type == VoicingType.DROP_3:
            return VoicingGenerator._generate_drop_3_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )
        elif voicing_type == VoicingType.DROP_2_4:
            return VoicingGenerator._generate_drop_2_4_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )
        elif voicing_type == VoicingType.SPREAD:
            return VoicingGenerator._generate_spread_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )
        else:  # OPEN
            return VoicingGenerator._generate_open_voicings(
                chord_tones_pc, voice_ranges, max_voicings
            )

    @staticmethod
    def _generate_close_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """Generate close position voicings (all voices within ~octave)"""
        voicings = []
        num_voices = len(voice_ranges)

        # Try different bass notes within range
        bass_range = voice_ranges[0]  # Lowest voice

        for bass_midi in range(bass_range.low, bass_range.high + 1):
            bass_pc = bass_midi % 12

            # Bass must be a chord tone
            if bass_pc not in chord_tones_pc:
                continue

            # Build voicing from bass up
            voicing = [bass_midi]
            current_pitch = bass_midi

            # Add remaining voices
            for i in range(1, num_voices):
                # Try to find next chord tone above current pitch
                found = False
                for offset in range(1, 13):  # Search up to octave
                    candidate = current_pitch + offset
                    candidate_pc = candidate % 12

                    # Check if it's a chord tone and within range
                    if (candidate_pc in chord_tones_pc and
                        voice_ranges[i].low <= candidate <= voice_ranges[i].high):
                        voicing.append(candidate)
                        current_pitch = candidate
                        found = True
                        break

                if not found:
                    # Can't complete this voicing
                    break

            # Add if we successfully created full voicing
            if len(voicing) == num_voices:
                # Check if it's actually close (span <= 12 semitones)
                if voicing[-1] - voicing[0] <= 12:
                    voicings.append(Voicing(voicing, VoicingType.CLOSE))

            if len(voicings) >= max_voicings:
                break

        return voicings

    @staticmethod
    def _generate_drop_2_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """
        Generate drop-2 voicings.

        Drop-2: Take close position, drop 2nd voice from top down an octave.
        This is THE most common big band voicing.
        """
        # First generate close voicings
        close_voicings = VoicingGenerator._generate_close_voicings(
            chord_tones_pc, voice_ranges, max_voicings * 2
        )

        drop_2_voicings = []

        for close_v in close_voicings:
            if len(close_v.pitches) < 3:
                continue

            # Drop 2nd from top down an octave
            voicing = close_v.pitches.copy()
            drop_idx = len(voicing) - 2  # Second from top
            voicing[drop_idx] -= 12

            # Check if still in range
            if voice_ranges[drop_idx].low <= voicing[drop_idx] <= voice_ranges[drop_idx].high:
                # Re-sort to maintain bass-to-soprano order
                sorted_voicing = sorted(voicing)
                drop_2_voicings.append(Voicing(sorted_voicing, VoicingType.DROP_2))

            if len(drop_2_voicings) >= max_voicings:
                break

        return drop_2_voicings

    @staticmethod
    def _generate_drop_3_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """Generate drop-3 voicings (drop 3rd voice from top down octave)"""
        close_voicings = VoicingGenerator._generate_close_voicings(
            chord_tones_pc, voice_ranges, max_voicings * 2
        )

        drop_3_voicings = []

        for close_v in close_voicings:
            if len(close_v.pitches) < 4:
                continue

            # Drop 3rd from top down an octave
            voicing = close_v.pitches.copy()
            drop_idx = len(voicing) - 3
            voicing[drop_idx] -= 12

            if voice_ranges[drop_idx].low <= voicing[drop_idx] <= voice_ranges[drop_idx].high:
                sorted_voicing = sorted(voicing)
                drop_3_voicings.append(Voicing(sorted_voicing, VoicingType.DROP_3))

            if len(drop_3_voicings) >= max_voicings:
                break

        return drop_3_voicings

    @staticmethod
    def _generate_drop_2_4_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """Generate drop-2-4 voicings (drop 2nd and 4th from top down octave)"""
        close_voicings = VoicingGenerator._generate_close_voicings(
            chord_tones_pc, voice_ranges, max_voicings * 2
        )

        drop_2_4_voicings = []

        for close_v in close_voicings:
            if len(close_v.pitches) < 4:
                continue

            # Drop 2nd and 4th from top down an octave
            voicing = close_v.pitches.copy()
            drop_indices = [len(voicing) - 2, len(voicing) - 4]

            valid = True
            for idx in drop_indices:
                if idx >= 0:
                    voicing[idx] -= 12
                    if not (voice_ranges[idx].low <= voicing[idx] <= voice_ranges[idx].high):
                        valid = False
                        break

            if valid:
                sorted_voicing = sorted(voicing)
                drop_2_4_voicings.append(Voicing(sorted_voicing, VoicingType.DROP_2_4))

            if len(drop_2_4_voicings) >= max_voicings:
                break

        return drop_2_4_voicings

    @staticmethod
    def _generate_spread_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """Generate spread voicings (wide spacing throughout)"""
        voicings = []
        num_voices = len(voice_ranges)

        # Spread: minimum 5 semitones between adjacent voices
        min_spacing = 5

        bass_range = voice_ranges[0]

        for bass_midi in range(bass_range.low, bass_range.high + 1):
            bass_pc = bass_midi % 12

            if bass_pc not in chord_tones_pc:
                continue

            # Use recursive backtracking to build spread voicing
            result = VoicingGenerator._build_spread_recursive(
                [bass_midi], 1, chord_tones_pc, voice_ranges, min_spacing
            )

            if result:
                voicings.append(Voicing(result, VoicingType.SPREAD))

            if len(voicings) >= max_voicings:
                break

        return voicings

    @staticmethod
    def _build_spread_recursive(
        current_voicing: List[int],
        voice_idx: int,
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        min_spacing: int
    ) -> Optional[List[int]]:
        """Recursively build spread voicing"""
        if voice_idx >= len(voice_ranges):
            return current_voicing

        last_pitch = current_voicing[-1]
        voice_range = voice_ranges[voice_idx]

        # Try pitches at least min_spacing above last pitch
        for candidate in range(last_pitch + min_spacing, voice_range.high + 1):
            if candidate % 12 in chord_tones_pc:
                result = VoicingGenerator._build_spread_recursive(
                    current_voicing + [candidate],
                    voice_idx + 1,
                    chord_tones_pc,
                    voice_ranges,
                    min_spacing
                )
                if result:
                    return result

        return None

    @staticmethod
    def _generate_open_voicings(
        chord_tones_pc: List[int],
        voice_ranges: List[VoiceRange],
        max_voicings: int
    ) -> List[Voicing]:
        """Generate open position voicings (spacing between close and spread)"""
        voicings = []
        num_voices = len(voice_ranges)

        # Open: spacing of 3-5 semitones between adjacent voices
        min_spacing = 3
        max_spacing = 8

        bass_range = voice_ranges[0]

        for bass_midi in range(bass_range.low, bass_range.high + 1):
            bass_pc = bass_midi % 12

            if bass_pc not in chord_tones_pc:
                continue

            voicing = [bass_midi]
            current_pitch = bass_midi

            for i in range(1, num_voices):
                found = False
                for offset in range(min_spacing, max_spacing + 1):
                    candidate = current_pitch + offset
                    candidate_pc = candidate % 12

                    if (candidate_pc in chord_tones_pc and
                        voice_ranges[i].low <= candidate <= voice_ranges[i].high):
                        voicing.append(candidate)
                        current_pitch = candidate
                        found = True
                        break

                if not found:
                    break

            if len(voicing) == num_voices:
                voicings.append(Voicing(voicing, VoicingType.OPEN))

            if len(voicings) >= max_voicings:
                break

        return voicings


# ============================================================================
# VOICE LEADING OPTIMIZER (MAIN CLASS)
# ============================================================================

class VoiceLeadingOptimizer:
    """
    Universal voice leading optimizer using dynamic programming.

    Finds optimal voicing sequence that minimizes voice movement between chords.
    """

    @staticmethod
    def calculate_voice_leading_distance(
        voicing1: List[int],
        voicing2: List[int],
        weights: Optional[List[float]] = None,
        strategy: MinimizationStrategy = MinimizationStrategy.TOTAL_MOTION
    ) -> float:
        """
        Calculate voice leading distance between two voicings.

        Args:
            voicing1: First voicing (MIDI pitches)
            voicing2: Second voicing (MIDI pitches)
            weights: Optional weights for each voice (default: equal weights)
            strategy: Minimization strategy

        Returns:
            Distance metric (lower is better)
        """
        if len(voicing1) != len(voicing2):
            raise ValueError("Voicings must have same number of voices")

        num_voices = len(voicing1)

        if weights is None:
            weights = [1.0] * num_voices

        if len(weights) != num_voices:
            raise ValueError("Weights must match number of voices")

        # Calculate based on strategy
        if strategy == MinimizationStrategy.TOTAL_MOTION:
            # Sum of weighted movements
            return sum(
                weights[i] * abs(voicing2[i] - voicing1[i])
                for i in range(num_voices)
            )

        elif strategy == MinimizationStrategy.MAX_LEAP:
            # Maximum single voice leap
            return max(abs(voicing2[i] - voicing1[i]) for i in range(num_voices))

        elif strategy == MinimizationStrategy.WEIGHTED:
            # Emphasize outer voices (bass and soprano)
            outer_weights = [1.5 if i in [0, num_voices-1] else 1.0 for i in range(num_voices)]
            return sum(
                outer_weights[i] * weights[i] * abs(voicing2[i] - voicing1[i])
                for i in range(num_voices)
            )

        elif strategy == MinimizationStrategy.COMMON_TONE:
            # Penalize non-common-tone movement
            # Lower distance if common tones are retained
            common_tones = sum(1 for i in range(num_voices) if voicing1[i] == voicing2[i])
            motion = sum(abs(voicing2[i] - voicing1[i]) for i in range(num_voices))
            # Reward common tones
            return motion - (common_tones * 2)

        else:
            # Default to total motion
            return sum(abs(voicing2[i] - voicing1[i]) for i in range(num_voices))

    @staticmethod
    def optimize_chord_sequence(
        chords: List[Dict],  # List of {root, quality} dicts
        num_voices: int = 4,
        voice_ranges: Optional[List[VoiceRange]] = None,
        voicing_types: List[VoicingType] = None,
        minimize: MinimizationStrategy = MinimizationStrategy.TOTAL_MOTION,
        weights: Optional[List[float]] = None
    ) -> OptimizationResult:
        """
        Find optimal voicings for chord sequence using dynamic programming.

        Algorithm:
        1. Generate all possible voicings for each chord (within ranges)
        2. Build graph: nodes = voicings, edges = voice leading distance
        3. Find shortest path through graph using DP

        Args:
            chords: List of chord dictionaries with 'root' and 'quality' keys
            num_voices: Number of voices
            voice_ranges: Range constraints for each voice (if None, use defaults)
            voicing_types: List of voicing types to consider (if None, use all)
            minimize: Minimization strategy
            weights: Optional weights for each voice

        Returns:
            OptimizationResult with optimal voicing sequence and metrics
        """
        if not chords:
            return OptimizationResult([], 0.0, 0.0, 0, [], 0)

        # Set default voice ranges if not provided
        if voice_ranges is None:
            voice_ranges = VoiceLeadingOptimizer._default_voice_ranges(num_voices)

        # Set default voicing types if not provided
        if voicing_types is None:
            voicing_types = [VoicingType.CLOSE, VoicingType.DROP_2]

        # Generate all possible voicings for each chord
        all_voicings = []
        for chord in chords:
            chord_voicings = []
            for v_type in voicing_types:
                voicings = VoicingGenerator.generate_all_voicings(
                    root=chord['root'],
                    quality=chord['quality'],
                    num_voices=num_voices,
                    voice_ranges=voice_ranges,
                    voicing_type=v_type,
                    max_voicings=200  # Limit per type
                )
                chord_voicings.extend(voicings)
            all_voicings.append(chord_voicings)

        # Use dynamic programming to find optimal path
        optimal_voicings = VoiceLeadingOptimizer._find_optimal_path_dp(
            all_voicings, minimize, weights
        )

        # Calculate metrics
        if len(optimal_voicings) < 2:
            return OptimizationResult(optimal_voicings, 0.0, 0.0, 0, [], 0)

        motion_per_step = []
        common_tones = 0

        for i in range(len(optimal_voicings) - 1):
            distance = VoiceLeadingOptimizer.calculate_voice_leading_distance(
                optimal_voicings[i].pitches,
                optimal_voicings[i + 1].pitches,
                weights,
                MinimizationStrategy.TOTAL_MOTION
            )
            motion_per_step.append(distance)

            # Count common tones
            for j in range(num_voices):
                if optimal_voicings[i].pitches[j] == optimal_voicings[i + 1].pitches[j]:
                    common_tones += 1

        total_motion = sum(motion_per_step)
        avg_motion = total_motion / len(motion_per_step)
        max_leap = max(
            max(abs(optimal_voicings[i].pitches[j] - optimal_voicings[i + 1].pitches[j])
                for j in range(num_voices))
            for i in range(len(optimal_voicings) - 1)
        )

        return OptimizationResult(
            voicings=optimal_voicings,
            total_motion=total_motion,
            avg_motion=avg_motion,
            max_leap=max_leap,
            motion_per_step=motion_per_step,
            common_tones_retained=common_tones
        )

    @staticmethod
    def _find_optimal_path_dp(
        voicing_options: List[List[Voicing]],
        strategy: MinimizationStrategy,
        weights: Optional[List[float]]
    ) -> List[Voicing]:
        """
        Find optimal path using dynamic programming.

        DP table: dp[chord_idx][voicing_idx] = (min_cost, prev_voicing_idx)

        Time complexity: O(n * m^2) where n=chords, m=voicings per chord
        """
        num_chords = len(voicing_options)

        if num_chords == 0:
            return []

        if num_chords == 1:
            # Just pick first voicing for single chord
            return [voicing_options[0][0]] if voicing_options[0] else []

        # Initialize DP table
        # dp[i] = dict mapping voicing_idx -> (min_cost, prev_voicing_idx)
        dp = [{}  for _ in range(num_chords)]

        # Base case: first chord (all voicings have cost 0)
        for i, voicing in enumerate(voicing_options[0]):
            dp[0][i] = (0.0, -1)

        # Fill DP table
        for chord_idx in range(1, num_chords):
            for curr_v_idx, curr_voicing in enumerate(voicing_options[chord_idx]):
                min_cost = float('inf')
                best_prev_idx = -1

                # Try all voicings from previous chord
                for prev_v_idx, prev_voicing in enumerate(voicing_options[chord_idx - 1]):
                    if prev_v_idx not in dp[chord_idx - 1]:
                        continue

                    prev_cost, _ = dp[chord_idx - 1][prev_v_idx]

                    # Calculate voice leading distance
                    distance = VoiceLeadingOptimizer.calculate_voice_leading_distance(
                        prev_voicing.pitches,
                        curr_voicing.pitches,
                        weights,
                        strategy
                    )

                    total_cost = prev_cost + distance

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_prev_idx = prev_v_idx

                dp[chord_idx][curr_v_idx] = (min_cost, best_prev_idx)

        # Backtrack to find optimal path
        # Find voicing with minimum cost in last chord
        last_chord_idx = num_chords - 1
        min_final_cost = float('inf')
        best_final_voicing_idx = -1

        for v_idx, (cost, _) in dp[last_chord_idx].items():
            if cost < min_final_cost:
                min_final_cost = cost
                best_final_voicing_idx = v_idx

        # Reconstruct path
        path_indices = []
        current_idx = best_final_voicing_idx

        for chord_idx in range(num_chords - 1, -1, -1):
            path_indices.append(current_idx)
            if chord_idx > 0:
                _, prev_idx = dp[chord_idx][current_idx]
                current_idx = prev_idx

        path_indices.reverse()

        # Convert indices to voicings
        optimal_path = [
            voicing_options[chord_idx][v_idx]
            for chord_idx, v_idx in enumerate(path_indices)
        ]

        return optimal_path

    @staticmethod
    def _default_voice_ranges(num_voices: int) -> List[VoiceRange]:
        """Create default voice ranges for generic use"""
        # Default ranges span 4 octaves from C2 (36) to C6 (84)
        # Distribute evenly across voices
        total_range = 48  # 4 octaves
        range_per_voice = total_range // num_voices

        ranges = []
        for i in range(num_voices):
            low = 36 + i * range_per_voice
            high = low + range_per_voice + 12  # Overlap with next voice
            comfortable_low = low + 3
            comfortable_high = high - 3

            ranges.append(VoiceRange(low, high, comfortable_low, comfortable_high))

        return ranges

    @staticmethod
    def apply_common_tone_retention(
        voicing1: List[int],
        voicing2_options: List[Voicing]
    ) -> Voicing:
        """
        From voicing2_options, choose one that maximizes common tones with voicing1.

        Common tone retention: if a pitch class appears in both chords,
        keep it in the same voice.

        Args:
            voicing1: Current voicing
            voicing2_options: List of possible next voicings

        Returns:
            Best voicing from options (most common tones retained)
        """
        if not voicing2_options:
            raise ValueError("No voicing options provided")

        best_voicing = voicing2_options[0]
        max_common_tones = 0

        for voicing2 in voicing2_options:
            # Count how many voices stay on same pitch
            common_tones = sum(
                1 for i in range(len(voicing1))
                if voicing1[i] == voicing2.pitches[i]
            )

            if common_tones > max_common_tones:
                max_common_tones = common_tones
                best_voicing = voicing2

        return best_voicing


# ============================================================================
# EXAMPLE USAGE AND VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("UNIVERSAL VOICE LEADING OPTIMIZER - EXAMPLES")
    print("=" * 80)

    # Example 1: ii-V-I progression (most common jazz progression)
    print("\n1. CLASSIC ii-V-I PROGRESSION (C major)")
    print("-" * 80)

    # Dm7 - G7 - Cmaj7
    chords = [
        {'root': 2, 'quality': 'min7'},   # Dm7
        {'root': 7, 'quality': 'dom7'},   # G7
        {'root': 0, 'quality': 'maj7'},   # Cmaj7
    ]

    # Define sax section ranges (SATB + Bari)
    sax_ranges = [
        VoiceRange(47, 76, 50, 70),   # Tenor 2 / Bari sax (lowest)
        VoiceRange(47, 76, 50, 70),   # Tenor 1
        VoiceRange(52, 81, 55, 76),   # Alto 2
        VoiceRange(52, 81, 55, 76),   # Alto 1 (highest)
    ]

    result = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords,
        num_voices=4,
        voice_ranges=sax_ranges,
        voicing_types=[VoicingType.DROP_2, VoicingType.CLOSE],
        minimize=MinimizationStrategy.TOTAL_MOTION
    )

    print(f"Optimal voicing sequence:")
    for i, voicing in enumerate(result.voicings):
        chord_name = ['Dm7', 'G7', 'Cmaj7'][i]
        print(f"  {chord_name}: {voicing.pitches} ({voicing.voicing_type.value})")

    print(f"\nMetrics:")
    print(f"  Total motion: {result.total_motion:.1f} semitones")
    print(f"  Average motion per change: {result.avg_motion:.1f} semitones")
    print(f"  Maximum leap: {result.max_leap} semitones")
    print(f"  Common tones retained: {result.common_tones_retained}")
    print(f"  Motion per step: {[f'{m:.1f}' for m in result.motion_per_step]}")

    # Example 2: I-IV-V-I progression
    print("\n2. I-IV-V-I PROGRESSION (F major)")
    print("-" * 80)

    chords_simple = [
        {'root': 5, 'quality': 'major'},   # F
        {'root': 10, 'quality': 'major'},  # Bb
        {'root': 0, 'quality': 'major'},   # C
        {'root': 5, 'quality': 'major'},   # F
    ]

    # String quartet ranges
    string_ranges = [
        VoiceRange(36, 72, 48, 60),   # Cello
        VoiceRange(48, 84, 55, 72),   # Viola
        VoiceRange(55, 91, 60, 84),   # Violin II
        VoiceRange(55, 96, 64, 88),   # Violin I
    ]

    result2 = VoiceLeadingOptimizer.optimize_chord_sequence(
        chords=chords_simple,
        num_voices=4,
        voice_ranges=string_ranges,
        voicing_types=[VoicingType.CLOSE],
        minimize=MinimizationStrategy.WEIGHTED  # Emphasize outer voices
    )

    print(f"Optimal voicing sequence (string quartet):")
    for i, voicing in enumerate(result2.voicings):
        chord_name = ['F', 'Bb', 'C', 'F'][i]
        print(f"  {chord_name}: {voicing.pitches}")

    print(f"\nMetrics:")
    print(f"  Total motion: {result2.total_motion:.1f} semitones")
    print(f"  Average motion: {result2.avg_motion:.1f} semitones")
    print(f"  Maximum leap: {result2.max_leap} semitones")

    # Example 3: Demonstrate common tone retention
    print("\n3. COMMON TONE RETENTION")
    print("-" * 80)

    # C major to Am (relative minor) - shares two common tones
    v1 = [48, 52, 55, 60]  # C E G C (Cmaj)
    v2_options = [
        Voicing([45, 52, 57, 60], VoicingType.CLOSE),  # A C E C (retains C, E)
        Voicing([45, 49, 52, 57], VoicingType.CLOSE),  # A C# E A (no retention)
    ]

    best = VoiceLeadingOptimizer.apply_common_tone_retention(v1, v2_options)
    print(f"Voicing 1 (Cmaj): {v1}")
    print(f"Best voicing 2 (Am): {best.pitches}")
    print(f"Common tones retained: C (52) and E (60)" if best.pitches == [45, 52, 57, 60] else "No common tones")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
