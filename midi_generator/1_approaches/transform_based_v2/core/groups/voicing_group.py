"""
Voicing and Cross-Track Relationship Types.

This module defines:
1. VoicingGroup: Voice-leading transforms (drop-2, drop-3, close, open)
2. CrossTrackRelationType: Types of relationships between tracks

These are used for cross-track analysis in the categorical layer.

Author: Architecture Refactor - Dosedo v2
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import IntEnum, auto

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class VoicingType(IntEnum):
    """Types of chord voicings."""
    CLOSE = 0           # All notes within one octave
    DROP_2 = 1          # Drop 2nd voice from top down an octave
    DROP_3 = 2          # Drop 3rd voice from top down an octave
    DROP_2_4 = 3        # Drop 2nd and 4th voices
    OPEN = 4            # Spread voicing (octave+ between adjacent notes)
    CLUSTER = 5         # Seconds/minor seconds (jazz clusters)
    SPREAD_TRIAD = 6    # Root + 10th + 5th (spread triad)
    FOUR_WAY_CLOSE = 7  # 4-way close (melody doubled at octave below)


class MotionType(IntEnum):
    """Types of voice motion between tracks."""
    PARALLEL = 0        # Same interval, same direction
    SIMILAR = 1         # Same direction, different intervals
    CONTRARY = 2        # Opposite directions
    OBLIQUE = 3         # One voice stationary


class CrossTrackRelationType(IntEnum):
    """
    Types of cross-track relationships.

    These define how two tracks relate to each other across time.
    """
    # Pitch relationships
    UNISON = 0              # Same pitches
    OCTAVE = 1              # Octave doubling
    PARALLEL_THIRDS = 2     # Parallel motion in thirds
    PARALLEL_SIXTHS = 3     # Parallel motion in sixths
    PARALLEL_FIFTHS = 4     # Parallel motion in fifths
    PARALLEL_FOURTHS = 5    # Parallel motion in fourths
    CONTRARY_MOTION = 6     # Opposite direction movement
    OBLIQUE_MOTION = 7      # One voice stationary

    # Rhythmic relationships
    RHYTHMIC_UNISON = 8     # Same rhythm
    RHYTHMIC_OFFSET = 9     # Offset by fixed amount
    RHYTHMIC_COMPLEMENT = 10  # Fills in the gaps
    HOCKET = 11             # Alternating notes

    # Structural relationships
    CALL_RESPONSE = 12      # Antecedent/consequent
    CANON = 13              # Delayed imitation
    INVERSION = 14          # Inverted melody
    RETROGRADE = 15         # Backwards version

    # Voicing relationships
    DROP_2_RELATION = 16    # Track is drop-2 of another
    DROP_3_RELATION = 17    # Track is drop-3 of another
    DOUBLING = 18           # Instrumental doubling
    HETEROPHONY = 19        # Same melody with embellishments


@dataclass
class VoicingTransform:
    """A voicing transformation."""
    name: str
    voicing_type: VoicingType
    description: str


class VoicingGroup:
    """
    Group of voicing transformations.

    These transforms modify how notes are distributed across octaves
    while preserving pitch class content.

    Common in jazz arranging:
    - Close voicing: all notes within one octave
    - Drop-2: second from top drops an octave (creates more open sound)
    - Drop-3: third from top drops an octave
    - Drop-2-4: both second and fourth drop (very open)
    """

    def __init__(self, device: str = 'cuda'):
        """Initialize voicing group."""
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'

        # Define transforms
        self.transforms = {
            VoicingType.CLOSE: VoicingTransform(
                'close', VoicingType.CLOSE,
                'Close position: all notes within one octave'
            ),
            VoicingType.DROP_2: VoicingTransform(
                'drop_2', VoicingType.DROP_2,
                'Drop-2: second note from top drops octave'
            ),
            VoicingType.DROP_3: VoicingTransform(
                'drop_3', VoicingType.DROP_3,
                'Drop-3: third note from top drops octave'
            ),
            VoicingType.DROP_2_4: VoicingTransform(
                'drop_2_4', VoicingType.DROP_2_4,
                'Drop-2-4: second and fourth notes drop octave'
            ),
            VoicingType.OPEN: VoicingTransform(
                'open', VoicingType.OPEN,
                'Open: spread across multiple octaves'
            ),
        }

    def apply_voicing(
        self,
        pitches: np.ndarray,
        voicing_type: VoicingType
    ) -> np.ndarray:
        """
        Apply voicing transformation to a set of pitches.

        Args:
            pitches: Array of MIDI pitches (sorted high to low)
            voicing_type: Type of voicing to apply

        Returns:
            Re-voiced pitches
        """
        if len(pitches) < 2:
            return pitches.copy()

        # Sort pitches high to low
        sorted_pitches = np.sort(pitches)[::-1]

        if voicing_type == VoicingType.CLOSE:
            return self._close_voicing(sorted_pitches)
        elif voicing_type == VoicingType.DROP_2:
            return self._drop_2_voicing(sorted_pitches)
        elif voicing_type == VoicingType.DROP_3:
            return self._drop_3_voicing(sorted_pitches)
        elif voicing_type == VoicingType.DROP_2_4:
            return self._drop_2_4_voicing(sorted_pitches)
        elif voicing_type == VoicingType.OPEN:
            return self._open_voicing(sorted_pitches)
        else:
            return sorted_pitches

    def _close_voicing(self, pitches: np.ndarray) -> np.ndarray:
        """
        Create close voicing: all notes within one octave.

        Stack notes from bottom, keeping within octave of root.
        """
        result = pitches.copy()
        root = pitches[-1]  # Lowest note

        for i in range(len(result) - 1):
            while result[i] - root >= 12:
                result[i] -= 12

        return np.sort(result)[::-1]

    def _drop_2_voicing(self, pitches: np.ndarray) -> np.ndarray:
        """
        Create drop-2 voicing: second from top drops an octave.

        Used heavily in big band writing.
        """
        if len(pitches) < 2:
            return pitches.copy()

        result = pitches.copy()
        result[1] -= 12  # Drop second from top

        return np.sort(result)[::-1]

    def _drop_3_voicing(self, pitches: np.ndarray) -> np.ndarray:
        """
        Create drop-3 voicing: third from top drops an octave.
        """
        if len(pitches) < 3:
            return pitches.copy()

        result = pitches.copy()
        result[2] -= 12  # Drop third from top

        return np.sort(result)[::-1]

    def _drop_2_4_voicing(self, pitches: np.ndarray) -> np.ndarray:
        """
        Create drop-2-4 voicing: second and fourth from top drop.

        Very open voicing used for 5+ note chords.
        """
        if len(pitches) < 4:
            return self._drop_2_voicing(pitches)

        result = pitches.copy()
        result[1] -= 12  # Drop second from top
        result[3] -= 12  # Drop fourth from top

        return np.sort(result)[::-1]

    def _open_voicing(self, pitches: np.ndarray) -> np.ndarray:
        """
        Create open voicing: spread across octaves.

        Alternates dropping notes by octave.
        """
        if len(pitches) < 2:
            return pitches.copy()

        result = pitches.copy()

        # Drop every other note
        for i in range(1, len(result), 2):
            result[i] -= 12

        return np.sort(result)[::-1]

    def detect_voicing_type(self, pitches: np.ndarray) -> VoicingType:
        """
        Detect the voicing type of a set of pitches.

        Args:
            pitches: Array of MIDI pitches

        Returns:
            Detected voicing type
        """
        if len(pitches) < 2:
            return VoicingType.CLOSE

        sorted_pitches = np.sort(pitches)[::-1]
        intervals = np.diff(sorted_pitches)

        # Check for close voicing (all intervals < 12)
        if np.all(np.abs(intervals) < 12):
            # Check for cluster (all intervals <= 3)
            if np.all(np.abs(intervals) <= 3):
                return VoicingType.CLUSTER
            return VoicingType.CLOSE

        # Check for drop voicings by looking at interval pattern
        if len(pitches) >= 3:
            # Drop-2 creates a large gap at position 1
            if len(intervals) >= 2 and intervals[0] >= 8:
                return VoicingType.DROP_2

            # Drop-3 creates a large gap at position 2
            if len(intervals) >= 3 and intervals[1] >= 8:
                return VoicingType.DROP_3

        return VoicingType.OPEN

    def find_voicing_relation(
        self,
        pitches_a: np.ndarray,
        pitches_b: np.ndarray
    ) -> Optional[VoicingType]:
        """
        Find if B is a voicing transformation of A.

        Args:
            pitches_a: Source pitches
            pitches_b: Target pitches

        Returns:
            VoicingType that transforms A to B, or None
        """
        # Must have same pitch classes
        pc_a = set(p % 12 for p in pitches_a)
        pc_b = set(p % 12 for p in pitches_b)

        if pc_a != pc_b:
            return None

        # Try each voicing type
        for vtype in VoicingType:
            transformed = self.apply_voicing(pitches_a, vtype)
            # Check if pitch classes match at same octaves (with tolerance)
            if len(transformed) == len(pitches_b):
                sorted_t = np.sort(transformed)
                sorted_b = np.sort(pitches_b)

                # Check if they match (allowing octave equivalence)
                if np.allclose((sorted_t - sorted_b) % 12, 0):
                    return vtype

        return None


# =============================================================================
# Cross-Track Relationship Detection
# =============================================================================

class CrossTrackAnalyzer:
    """
    Analyzes relationships between pairs of tracks.

    Used by the categorical cross-track layer to find natural transformations.
    """

    def __init__(self, device: str = 'cuda'):
        """Initialize analyzer."""
        self.device = device if HAS_TORCH and torch.cuda.is_available() else 'cpu'
        self.voicing_group = VoicingGroup(device)

    def detect_pitch_relationship(
        self,
        pitches_a: np.ndarray,
        pitches_b: np.ndarray
    ) -> Tuple[Optional[CrossTrackRelationType], Optional[int]]:
        """
        Detect pitch relationship between two pitch sequences.

        Args:
            pitches_a: Pitches from track A
            pitches_b: Pitches from track B (same length)

        Returns:
            (relationship_type, parameter) or (None, None)
        """
        if len(pitches_a) != len(pitches_b) or len(pitches_a) == 0:
            return None, None

        # Compute intervals
        intervals = pitches_b - pitches_a

        # Check for constant interval
        if np.all(intervals == intervals[0]):
            interval = int(intervals[0])

            if interval == 0:
                return CrossTrackRelationType.UNISON, 0
            elif interval % 12 == 0:
                return CrossTrackRelationType.OCTAVE, interval // 12
            elif interval % 12 in [3, 4]:  # Minor/major third
                return CrossTrackRelationType.PARALLEL_THIRDS, interval
            elif interval % 12 in [8, 9]:  # Minor/major sixth
                return CrossTrackRelationType.PARALLEL_SIXTHS, interval
            elif interval % 12 == 7:  # Perfect fifth
                return CrossTrackRelationType.PARALLEL_FIFTHS, interval
            elif interval % 12 == 5:  # Perfect fourth
                return CrossTrackRelationType.PARALLEL_FOURTHS, interval

        # Check for contrary motion
        if len(pitches_a) >= 2:
            motion_a = np.diff(pitches_a)
            motion_b = np.diff(pitches_b)

            # Contrary: opposite signs
            contrary_count = np.sum(motion_a * motion_b < 0)
            if contrary_count / len(motion_a) > 0.8:
                return CrossTrackRelationType.CONTRARY_MOTION, None

            # Oblique: one is stationary
            if np.sum(motion_a == 0) / len(motion_a) > 0.8:
                return CrossTrackRelationType.OBLIQUE_MOTION, None
            if np.sum(motion_b == 0) / len(motion_b) > 0.8:
                return CrossTrackRelationType.OBLIQUE_MOTION, None

        return None, None

    def detect_rhythm_relationship(
        self,
        rhythm_a: np.ndarray,
        rhythm_b: np.ndarray
    ) -> Tuple[Optional[CrossTrackRelationType], Optional[int]]:
        """
        Detect rhythmic relationship between two rhythm patterns.

        Args:
            rhythm_a: Binary onset pattern from track A
            rhythm_b: Binary onset pattern from track B

        Returns:
            (relationship_type, parameter) or (None, None)
        """
        if len(rhythm_a) != len(rhythm_b):
            return None, None

        # Check for unison
        if np.array_equal(rhythm_a, rhythm_b):
            return CrossTrackRelationType.RHYTHMIC_UNISON, 0

        # Check for offset (shifted version)
        for offset in range(1, min(32, len(rhythm_a))):
            shifted = np.roll(rhythm_a, offset)
            if np.array_equal(shifted, rhythm_b):
                return CrossTrackRelationType.RHYTHMIC_OFFSET, offset

        # Check for complement (fills gaps)
        combined = rhythm_a + rhythm_b
        if np.sum(combined > 1) == 0:  # No overlaps
            # They complement each other
            coverage = np.sum(combined > 0) / len(combined)
            if coverage > 0.8:
                return CrossTrackRelationType.RHYTHMIC_COMPLEMENT, None

        # Check for hocket (alternating)
        alternating = np.zeros_like(rhythm_a)
        alternating[::2] = 1
        corr = np.corrcoef(rhythm_a.flatten(), alternating.flatten())
        if corr.shape == (2, 2) and corr[0, 1] > 0.7:
            anti_alt = 1 - alternating
            corr2 = np.corrcoef(rhythm_b.flatten(), anti_alt.flatten())
            if corr2.shape == (2, 2) and corr2[0, 1] > 0.7:
                return CrossTrackRelationType.HOCKET, None

        return None, None

    def detect_melodic_relationship(
        self,
        contour_a: np.ndarray,
        contour_b: np.ndarray
    ) -> Tuple[Optional[CrossTrackRelationType], Optional[int]]:
        """
        Detect melodic relationship based on contours.

        Args:
            contour_a: Pitch intervals from track A
            contour_b: Pitch intervals from track B

        Returns:
            (relationship_type, parameter) or (None, None)
        """
        if len(contour_a) != len(contour_b) or len(contour_a) == 0:
            return None, None

        # Check for inversion
        if np.allclose(contour_a, -contour_b):
            return CrossTrackRelationType.INVERSION, None

        # Check for retrograde
        if np.allclose(contour_a, contour_b[::-1]):
            return CrossTrackRelationType.RETROGRADE, None

        return None, None

    def find_all_relationships(
        self,
        track_a_data: dict,
        track_b_data: dict
    ) -> List[Tuple[CrossTrackRelationType, Optional[int]]]:
        """
        Find all relationships between two tracks.

        Args:
            track_a_data: Dict with 'pitches', 'rhythm', 'contour'
            track_b_data: Dict with 'pitches', 'rhythm', 'contour'

        Returns:
            List of (relationship_type, parameter) tuples
        """
        relationships = []

        # Pitch relationships
        if 'pitches' in track_a_data and 'pitches' in track_b_data:
            rel, param = self.detect_pitch_relationship(
                track_a_data['pitches'],
                track_b_data['pitches']
            )
            if rel is not None:
                relationships.append((rel, param))

        # Rhythm relationships
        if 'rhythm' in track_a_data and 'rhythm' in track_b_data:
            rel, param = self.detect_rhythm_relationship(
                track_a_data['rhythm'],
                track_b_data['rhythm']
            )
            if rel is not None:
                relationships.append((rel, param))

        # Melodic relationships
        if 'contour' in track_a_data and 'contour' in track_b_data:
            rel, param = self.detect_melodic_relationship(
                track_a_data['contour'],
                track_b_data['contour']
            )
            if rel is not None:
                relationships.append((rel, param))

        return relationships


# =============================================================================
# Convenience Functions
# =============================================================================

def relationship_name(rel_type: CrossTrackRelationType) -> str:
    """Get human-readable name for relationship type."""
    names = {
        CrossTrackRelationType.UNISON: "Unison",
        CrossTrackRelationType.OCTAVE: "Octave doubling",
        CrossTrackRelationType.PARALLEL_THIRDS: "Parallel 3rds",
        CrossTrackRelationType.PARALLEL_SIXTHS: "Parallel 6ths",
        CrossTrackRelationType.PARALLEL_FIFTHS: "Parallel 5ths",
        CrossTrackRelationType.PARALLEL_FOURTHS: "Parallel 4ths",
        CrossTrackRelationType.CONTRARY_MOTION: "Contrary motion",
        CrossTrackRelationType.OBLIQUE_MOTION: "Oblique motion",
        CrossTrackRelationType.RHYTHMIC_UNISON: "Rhythmic unison",
        CrossTrackRelationType.RHYTHMIC_OFFSET: "Rhythmic offset",
        CrossTrackRelationType.RHYTHMIC_COMPLEMENT: "Rhythmic complement",
        CrossTrackRelationType.HOCKET: "Hocket",
        CrossTrackRelationType.CALL_RESPONSE: "Call & response",
        CrossTrackRelationType.CANON: "Canon",
        CrossTrackRelationType.INVERSION: "Melodic inversion",
        CrossTrackRelationType.RETROGRADE: "Retrograde",
        CrossTrackRelationType.DROP_2_RELATION: "Drop-2 voicing",
        CrossTrackRelationType.DROP_3_RELATION: "Drop-3 voicing",
        CrossTrackRelationType.DOUBLING: "Doubling",
        CrossTrackRelationType.HETEROPHONY: "Heterophony",
    }
    return names.get(rel_type, str(rel_type))
