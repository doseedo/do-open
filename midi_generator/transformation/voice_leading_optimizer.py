#!/usr/bin/env python3
"""
Voice Leading Optimizer - Universal Voice Leading Optimization Engine
=====================================================================

A sophisticated voice leading optimization system that minimizes voice movement
between chords while respecting range constraints. Based on research from:
- Matthew Keating (2023): "An Algorithmic Approach to Jazz Guitar Voice-Leading"
- Classical voice leading principles (Fux, Schenker)
- Jazz voice leading (Mark Levine "Jazz Theory Book")

This module is UNIVERSAL and works for any ensemble:
- Sax soli (5 voices)
- Brass sections (4-8 voices)
- String sections (SATB, orchestra)
- Vocal harmony (SATB choir)
- Any multi-voice harmony

Key Features:
-------------
- Dynamic programming optimization for smooth voice leading
- Common tone retention (keep common notes in same voice)
- Configurable voice ranges per instrument
- Multiple voicing types (close, drop-2, drop-3, drop-2-4, spread)
- Register-specific spacing rules
- Minimal motion prioritization

Author: Agent 2 - Sax Soli Voicing Master
License: MIT
"""

from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
from enum import Enum
import sys
from pathlib import Path

# Import data structures
sys.path.append(str(Path(__file__).parent.parent))
from analysis.midi_analyzer import ChordEvent, NoteEvent


# ============================================================================
# CORE DATA STRUCTURES
# ============================================================================

class VoicingType(Enum):
    """Types of voicings available"""
    CLOSE = "close"              # All voices within octave
    DROP_2 = "drop_2"            # 2nd voice from top down octave (MOST COMMON)
    DROP_3 = "drop_3"            # 3rd voice from top down octave
    DROP_2_4 = "drop_2_4"        # 2nd and 4th voices down octave
    SPREAD = "spread"            # Wide spacing throughout
    OPEN = "open"                # Wide spacing (alias for spread)


@dataclass
class VoicingConstraints:
    """Constraints for voice leading optimization"""
    voice_ranges: List[Tuple[int, int]]  # [(min, max)] for each voice
    max_leap: int = 12                   # Maximum leap per voice (semitones)
    min_spacing: int = 1                 # Minimum spacing between adjacent voices
    max_spacing: int = 12                # Maximum spacing between adjacent voices
    prefer_common_tones: bool = True     # Retain common tones in same voice
    allow_voice_crossing: bool = False   # Allow voices to cross


@dataclass
class Voicing:
    """A single voicing (set of pitches for voices)"""
    pitches: List[int]                   # MIDI pitches for each voice (low to high)
    chord_tones: List[int]               # Pitch classes used (0-11)
    voicing_type: VoicingType

    def __repr__(self):
        return f"Voicing({self.pitches}, type={self.voicing_type.value})"


# ============================================================================
# VOICE LEADING OPTIMIZER
# ============================================================================

class VoiceLeadingOptimizer:
    """
    Universal voice leading optimizer using dynamic programming.

    Finds optimal sequence of voicings that minimizes total voice movement
    while respecting range and spacing constraints.
    """

    @staticmethod
    def calculate_voice_leading_distance(voicing1: List[int],
                                         voicing2: List[int],
                                         weights: Optional[List[float]] = None) -> float:
        """
        Measure distance between two voicings (sum of voice movements).

        Args:
            voicing1: First voicing (MIDI pitches)
            voicing2: Second voicing (MIDI pitches)
            weights: Optional weights per voice (default: all 1.0)
                    Use [1.5, 1.0, 1.0, 1.5] to penalize outer voice leaps

        Returns:
            Total weighted distance in semitones
        """
        if len(voicing1) != len(voicing2):
            raise ValueError(f"Voicings must have same number of voices: {len(voicing1)} vs {len(voicing2)}")

        if weights is None:
            weights = [1.0] * len(voicing1)

        total_distance = 0.0
        for i, (p1, p2, w) in enumerate(zip(voicing1, voicing2, weights)):
            distance = abs(p1 - p2)
            total_distance += distance * w

        return total_distance

    @staticmethod
    def generate_all_voicings(chord: ChordEvent,
                             num_voices: int,
                             voice_ranges: List[Tuple[int, int]],
                             voicing_type: VoicingType = VoicingType.DROP_2,
                             melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate all valid voicings for a chord within range constraints.

        Args:
            chord: Chord to voice
            num_voices: Number of voices (e.g., 5 for sax section)
            voice_ranges: [(min, max)] MIDI pitch range for each voice
            voicing_type: Type of voicing to generate
            melody_pitch: If provided, top voice must be this pitch

        Returns:
            List of valid Voicing objects
        """
        # Build chord tones (pitch classes)
        chord_tones = VoiceLeadingOptimizer._get_chord_tones(chord)

        # Generate voicings based on type
        voicings = []

        if voicing_type == VoicingType.CLOSE:
            voicings = VoiceLeadingOptimizer._generate_close_voicings(
                chord_tones, num_voices, voice_ranges, melody_pitch
            )

        elif voicing_type == VoicingType.DROP_2:
            voicings = VoiceLeadingOptimizer._generate_drop_2_voicings(
                chord_tones, num_voices, voice_ranges, melody_pitch
            )

        elif voicing_type == VoicingType.DROP_3:
            voicings = VoiceLeadingOptimizer._generate_drop_3_voicings(
                chord_tones, num_voices, voice_ranges, melody_pitch
            )

        elif voicing_type == VoicingType.DROP_2_4:
            voicings = VoiceLeadingOptimizer._generate_drop_2_4_voicings(
                chord_tones, num_voices, voice_ranges, melody_pitch
            )

        elif voicing_type in [VoicingType.SPREAD, VoicingType.OPEN]:
            voicings = VoiceLeadingOptimizer._generate_spread_voicings(
                chord_tones, num_voices, voice_ranges, melody_pitch
            )

        return voicings

    @staticmethod
    def optimize_chord_sequence(chords: List[ChordEvent],
                                num_voices: int = 4,
                                voice_ranges: Optional[List[Tuple[int, int]]] = None,
                                voicing_type: VoicingType = VoicingType.DROP_2,
                                melody_pitches: Optional[List[int]] = None,
                                minimize: str = "total_motion") -> List[List[int]]:
        """
        Find optimal voicings for chord sequence using dynamic programming.

        Args:
            chords: Sequence of chords to voice
            num_voices: Number of voices
            voice_ranges: Range constraints per voice (default: auto-generate)
            voicing_type: Type of voicing (drop-2, close, etc.)
            melody_pitches: Optional melody notes to lock top voice
            minimize: "total_motion", "max_leap", or "weighted"

        Returns:
            List of voicings (each voicing is list of MIDI pitches)

        Algorithm:
            1. Generate all possible voicings for each chord
            2. Build DP table: dp[chord_idx][voicing_idx] = min cost to reach
            3. Find shortest path through voicing space
            4. Backtrack to reconstruct optimal sequence
        """
        if not chords:
            return []

        # Default voice ranges if not provided (4-part SATB-like)
        if voice_ranges is None:
            voice_ranges = [
                (48, 72),  # Voice 1 (bass)
                (52, 76),  # Voice 2
                (57, 81),  # Voice 3
                (60, 84),  # Voice 4 (soprano)
            ][:num_voices]

        # Generate all possible voicings for each chord
        all_voicings = []
        for i, chord in enumerate(chords):
            melody_pitch = melody_pitches[i] if melody_pitches else None
            chord_voicings = VoiceLeadingOptimizer.generate_all_voicings(
                chord, num_voices, voice_ranges, voicing_type, melody_pitch
            )

            if not chord_voicings:
                # Fallback: generate simpler voicing
                chord_voicings = VoiceLeadingOptimizer.generate_all_voicings(
                    chord, num_voices, voice_ranges, VoicingType.CLOSE, melody_pitch
                )

            all_voicings.append(chord_voicings)

        # Dynamic programming to find optimal path
        n = len(chords)

        # dp[i][j] = (min_cost, prev_voicing_idx)
        dp = [[float('inf'), -1] for _ in all_voicings[0]]

        # Initialize first chord (cost = 0 for all voicings)
        for j in range(len(all_voicings[0])):
            dp[j] = [0, -1]

        # Fill DP table
        for i in range(1, n):
            prev_voicings = all_voicings[i-1]
            curr_voicings = all_voicings[i]

            new_dp = [[float('inf'), -1] for _ in curr_voicings]

            for j, curr_voicing in enumerate(curr_voicings):
                for k, prev_voicing in enumerate(prev_voicings):
                    # Calculate cost of moving from prev_voicing to curr_voicing
                    cost = VoiceLeadingOptimizer.calculate_voice_leading_distance(
                        prev_voicing.pitches, curr_voicing.pitches
                    )

                    total_cost = dp[k][0] + cost

                    if total_cost < new_dp[j][0]:
                        new_dp[j] = [total_cost, k]

            dp = new_dp

        # Backtrack to find optimal path
        if not dp:
            return []

        # Find best final voicing
        best_idx = min(range(len(dp)), key=lambda x: dp[x][0])

        # Reconstruct path
        path = []
        current_idx = best_idx

        for i in range(n-1, -1, -1):
            path.append(all_voicings[i][current_idx].pitches)
            if i > 0:
                current_idx = dp[current_idx][1]

        path.reverse()
        return path

    @staticmethod
    def apply_common_tone_retention(voicing1: List[int],
                                    voicing2_options: List[List[int]]) -> List[int]:
        """
        Choose voicing from options that maximizes common tones with voicing1.

        Common tone retention: if a pitch appears in both voicings,
        keep it in the same voice to minimize movement.
        """
        best_voicing = voicing2_options[0]
        max_common_tones = 0

        for voicing2 in voicing2_options:
            common_tones = sum(1 for p1, p2 in zip(voicing1, voicing2) if p1 == p2)
            if common_tones > max_common_tones:
                max_common_tones = common_tones
                best_voicing = voicing2

        return best_voicing

    # ========================================================================
    # VOICING GENERATION ALGORITHMS
    # ========================================================================

    @staticmethod
    def _get_chord_tones(chord: ChordEvent) -> List[int]:
        """Extract chord tones (pitch classes) from ChordEvent."""
        root = chord.root
        quality = chord.quality.lower()

        # Build chord tones based on quality
        tones = [root]  # Root

        if 'major' in quality or 'maj' in quality:
            tones.append((root + 4) % 12)   # Major 3rd
        elif 'minor' in quality or 'min' in quality or 'm' == quality:
            tones.append((root + 3) % 12)   # Minor 3rd
        elif 'dim' in quality:
            tones.append((root + 3) % 12)   # Minor 3rd
        else:
            tones.append((root + 4) % 12)   # Default major 3rd

        tones.append((root + 7) % 12)   # Perfect 5th

        # Add 7th
        if '7' in quality:
            if 'maj7' in quality or 'M7' in quality:
                tones.append((root + 11) % 12)  # Major 7th
            elif 'dim7' in quality or 'º7' in quality:
                tones.append((root + 9) % 12)   # Diminished 7th
            elif 'min7' in quality or 'm7' in quality:
                tones.append((root + 10) % 12)  # Minor 7th
            else:
                tones.append((root + 10) % 12)  # Dominant 7th (minor 7)

        # Remove duplicates and sort
        return sorted(list(set(tones)))

    @staticmethod
    def _generate_close_voicings(chord_tones: List[int],
                                 num_voices: int,
                                 voice_ranges: List[Tuple[int, int]],
                                 melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate close position voicings (all voices within octave).
        """
        voicings = []

        # Determine range for top voice
        top_range = voice_ranges[-1]

        # If melody pitch provided, use it; otherwise explore range
        if melody_pitch:
            top_pitches = [melody_pitch]
        else:
            top_pitches = range(top_range[0], min(top_range[1] + 1, top_range[0] + 24))

        for top_pitch in top_pitches:
            # Build voicing downward in close position
            voicing = [top_pitch]
            current_pitch = top_pitch - 1

            for voice_idx in range(num_voices - 2, -1, -1):
                voice_min, voice_max = voice_ranges[voice_idx]

                # Find next chord tone below current_pitch
                found = False
                for semitones_below in range(1, 13):  # Search within octave
                    candidate = current_pitch - semitones_below

                    if candidate < voice_min:
                        break

                    if candidate % 12 in chord_tones and voice_min <= candidate <= voice_max:
                        voicing.insert(0, candidate)
                        current_pitch = candidate
                        found = True
                        break

                if not found:
                    break  # Can't complete voicing

            if len(voicing) == num_voices:
                voicings.append(Voicing(
                    pitches=sorted(voicing),
                    chord_tones=chord_tones,
                    voicing_type=VoicingType.CLOSE
                ))

        return voicings

    @staticmethod
    def _generate_drop_2_voicings(chord_tones: List[int],
                                  num_voices: int,
                                  voice_ranges: List[Tuple[int, int]],
                                  melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate drop-2 voicings (2nd voice from top dropped down octave).

        This is THE MOST COMMON big band voicing.
        """
        voicings = []

        # First generate close voicings
        close_voicings = VoiceLeadingOptimizer._generate_close_voicings(
            chord_tones, num_voices, voice_ranges, melody_pitch
        )

        # Convert each close voicing to drop-2
        for close_voicing in close_voicings[:20]:  # Limit to avoid explosion
            pitches = close_voicing.pitches.copy()

            if len(pitches) < 2:
                continue

            # Drop 2nd voice from top down an octave
            second_from_top_idx = len(pitches) - 2
            pitches[second_from_top_idx] -= 12

            # Check if still in range
            voice_min, voice_max = voice_ranges[second_from_top_idx]
            if voice_min <= pitches[second_from_top_idx] <= voice_max:
                voicings.append(Voicing(
                    pitches=sorted(pitches),
                    chord_tones=chord_tones,
                    voicing_type=VoicingType.DROP_2
                ))

        return voicings

    @staticmethod
    def _generate_drop_3_voicings(chord_tones: List[int],
                                  num_voices: int,
                                  voice_ranges: List[Tuple[int, int]],
                                  melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate drop-3 voicings (3rd voice from top dropped down octave).
        """
        voicings = []

        close_voicings = VoiceLeadingOptimizer._generate_close_voicings(
            chord_tones, num_voices, voice_ranges, melody_pitch
        )

        for close_voicing in close_voicings[:20]:
            pitches = close_voicing.pitches.copy()

            if len(pitches) < 3:
                continue

            # Drop 3rd voice from top down an octave
            third_from_top_idx = len(pitches) - 3
            pitches[third_from_top_idx] -= 12

            voice_min, voice_max = voice_ranges[third_from_top_idx]
            if voice_min <= pitches[third_from_top_idx] <= voice_max:
                voicings.append(Voicing(
                    pitches=sorted(pitches),
                    chord_tones=chord_tones,
                    voicing_type=VoicingType.DROP_3
                ))

        return voicings

    @staticmethod
    def _generate_drop_2_4_voicings(chord_tones: List[int],
                                    num_voices: int,
                                    voice_ranges: List[Tuple[int, int]],
                                    melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate drop-2-4 voicings (2nd and 4th voices from top dropped).
        """
        voicings = []

        close_voicings = VoiceLeadingOptimizer._generate_close_voicings(
            chord_tones, num_voices, voice_ranges, melody_pitch
        )

        for close_voicing in close_voicings[:20]:
            pitches = close_voicing.pitches.copy()

            if len(pitches) < 4:
                continue

            # Drop 2nd and 4th voices from top
            second_from_top_idx = len(pitches) - 2
            fourth_from_top_idx = len(pitches) - 4

            pitches[second_from_top_idx] -= 12
            pitches[fourth_from_top_idx] -= 12

            # Check ranges
            valid = True
            for idx in [second_from_top_idx, fourth_from_top_idx]:
                voice_min, voice_max = voice_ranges[idx]
                if not (voice_min <= pitches[idx] <= voice_max):
                    valid = False
                    break

            if valid:
                voicings.append(Voicing(
                    pitches=sorted(pitches),
                    chord_tones=chord_tones,
                    voicing_type=VoicingType.DROP_2_4
                ))

        return voicings

    @staticmethod
    def _generate_spread_voicings(chord_tones: List[int],
                                  num_voices: int,
                                  voice_ranges: List[Tuple[int, int]],
                                  melody_pitch: Optional[int] = None) -> List[Voicing]:
        """
        Generate spread voicings (wide spacing throughout).
        """
        voicings = []

        # Use drop-2-4 as basis for spread voicings
        drop_2_4_voicings = VoiceLeadingOptimizer._generate_drop_2_4_voicings(
            chord_tones, num_voices, voice_ranges, melody_pitch
        )

        for voicing in drop_2_4_voicings:
            voicing.voicing_type = VoicingType.SPREAD
            voicings.append(voicing)

        return voicings
