#!/usr/bin/env python3
"""
Motif Library and Database System
==================================

This module implements automatic extraction, tagging, storage, and retrieval
of musical motifs from MIDI files. It creates a searchable database of melodic
fragments that can be combined and transformed for composition.

Features:
- Automatic motif extraction from famous pieces
- Multi-dimensional tagging (emotion, genre, composer, era)
- Similarity search and motif clustering
- Motif combination and transition algorithms
- Database persistence (JSON, SQLite)
- Statistical analysis of motif characteristics

Research foundations:
- Motivic analysis (Schoenberg, Rudolph Réti)
- Musical information retrieval
- Melodic similarity (Typke et al., 2005)
- Thematic cataloging (Barlow & Morgenstern)

Author: Agent 9 - ML Integration & Pattern Discovery
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict, Counter
from enum import Enum
import json
import sqlite3
from pathlib import Path
import math
import hashlib

try:
    import numpy as np
    from scipy.spatial.distance import euclidean, cosine
    from sklearn.metrics.pairwise import cosine_similarity
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy/scipy not available. Install for enhanced features.")


class EmotionTag(Enum):
    """Emotional characteristics of motifs."""
    JOYFUL = "joyful"
    SAD = "sad"
    TRIUMPHANT = "triumphant"
    MYSTERIOUS = "mysterious"
    PEACEFUL = "peaceful"
    AGITATED = "agitated"
    ROMANTIC = "romantic"
    DARK = "dark"
    PLAYFUL = "playful"
    NOSTALGIC = "nostalgic"
    HEROIC = "heroic"
    TENDER = "tender"


class ContourType(Enum):
    """Melodic contour types."""
    ASCENDING = "ascending"
    DESCENDING = "descending"
    ARCH = "arch"  # Up then down
    VALLEY = "valley"  # Down then up
    WAVE = "wave"  # Oscillating
    STATIC = "static"  # Mostly repeated notes
    ANGULAR = "angular"  # Large leaps


@dataclass
class Motif:
    """
    Represents a musical motif (short melodic fragment).

    Attributes:
        id: Unique identifier
        notes: Pitch sequence (MIDI note numbers)
        intervals: Interval sequence (transposition-invariant)
        rhythm: Duration sequence
        contour: Melodic contour pattern
        source: Source piece/composer
        tags: Multiple categorization tags
    """
    id: str
    notes: List[int]
    intervals: List[int]
    rhythm: List[float]
    contour: List[int]  # -1, 0, 1 for down, same, up

    # Metadata
    source: str = ""
    composer: str = ""
    genre: str = ""
    era: str = ""
    emotion_tags: List[str] = field(default_factory=list)
    contour_type: str = ""

    # Statistics
    length: int = 0
    pitch_range: int = 0
    avg_interval: float = 0.0
    max_leap: int = 0

    # Usage tracking
    usage_count: int = 0
    rating: float = 0.0

    def __post_init__(self):
        """Compute derived attributes."""
        if self.length == 0:
            self.length = len(self.notes)

        if self.pitch_range == 0 and self.notes:
            self.pitch_range = max(self.notes) - min(self.notes)

        if self.avg_interval == 0 and self.intervals:
            self.avg_interval = sum(abs(i) for i in self.intervals) / len(self.intervals)

        if self.max_leap == 0 and self.intervals:
            self.max_leap = max(abs(i) for i in self.intervals)

        if not self.contour_type:
            self.contour_type = self._classify_contour()

        # Generate ID if not provided
        if not self.id:
            self.id = self._generate_id()

    def _classify_contour(self) -> str:
        """Classify the overall contour shape."""
        if not self.contour:
            return ContourType.STATIC.value

        ups = sum(1 for c in self.contour if c > 0)
        downs = sum(1 for c in self.contour if c < 0)
        total = len(self.contour)

        if ups > total * 0.7:
            return ContourType.ASCENDING.value
        elif downs > total * 0.7:
            return ContourType.DESCENDING.value
        elif self.max_leap > 5:
            return ContourType.ANGULAR.value

        # Check for arch/valley
        mid = len(self.contour) // 2
        first_half_up = sum(self.contour[:mid]) > 0
        second_half_down = sum(self.contour[mid:]) < 0

        if first_half_up and second_half_down:
            return ContourType.ARCH.value
        elif sum(self.contour[:mid]) < 0 and sum(self.contour[mid:]) > 0:
            return ContourType.VALLEY.value

        return ContourType.WAVE.value

    def _generate_id(self) -> str:
        """Generate unique ID from motif content."""
        content = f"{self.intervals}{self.rhythm}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def transpose(self, semitones: int) -> 'Motif':
        """Create transposed version of motif."""
        transposed_notes = [n + semitones for n in self.notes]

        return Motif(
            id=self.id + f"_T{semitones:+d}",
            notes=transposed_notes,
            intervals=self.intervals.copy(),
            rhythm=self.rhythm.copy(),
            contour=self.contour.copy(),
            source=self.source,
            composer=self.composer,
            genre=self.genre,
            era=self.era,
            emotion_tags=self.emotion_tags.copy(),
        )

    def retrograde(self) -> 'Motif':
        """Create retrograde (reversed) version."""
        return Motif(
            id=self.id + "_R",
            notes=list(reversed(self.notes)),
            intervals=list(reversed([-i for i in self.intervals])),
            rhythm=list(reversed(self.rhythm)),
            contour=list(reversed([-c for c in self.contour])),
            source=self.source,
            composer=self.composer,
        )

    def inversion(self) -> 'Motif':
        """Create inverted version (intervals flipped)."""
        inverted_intervals = [-i for i in self.intervals]
        inverted_notes = [self.notes[0]]

        for interval in inverted_intervals:
            inverted_notes.append(inverted_notes[-1] + interval)

        return Motif(
            id=self.id + "_I",
            notes=inverted_notes,
            intervals=inverted_intervals,
            rhythm=self.rhythm.copy(),
            contour=[-c for c in self.contour],
            source=self.source,
            composer=self.composer,
        )

    def augmentation(self, factor: float = 2.0) -> 'Motif':
        """Create augmented version (rhythmically slower)."""
        return Motif(
            id=self.id + f"_Aug{factor}",
            notes=self.notes.copy(),
            intervals=self.intervals.copy(),
            rhythm=[r * factor for r in self.rhythm],
            contour=self.contour.copy(),
            source=self.source,
        )

    def diminution(self, factor: float = 0.5) -> 'Motif':
        """Create diminuted version (rhythmically faster)."""
        return self.augmentation(factor)


class MotifExtractor:
    """
    Extract motifs from musical sequences using various heuristics.

    Identifies salient melodic fragments based on:
    - Repetition
    - Melodic distinctiveness
    - Rhythmic interest
    - Position in phrase structure
    """

    def __init__(self, min_length: int = 3, max_length: int = 12):
        """
        Initialize motif extractor.

        Args:
            min_length: Minimum motif length in notes
            max_length: Maximum motif length in notes
        """
        self.min_length = min_length
        self.max_length = max_length

    def extract_from_sequence(self, pitches: List[int],
                              durations: List[float] = None,
                              **metadata) -> List[Motif]:
        """
        Extract motifs from a melodic sequence.

        Args:
            pitches: MIDI pitch sequence
            durations: Optional duration sequence
            **metadata: Additional metadata (source, composer, etc.)

        Returns:
            List of extracted Motif objects
        """
        motifs = []

        if durations is None:
            durations = [1.0] * len(pitches)

        # Find repeated subsequences
        repeated_motifs = self._find_repeated_subsequences(pitches, durations)
        motifs.extend(repeated_motifs)

        # Find rhythmically distinctive motifs
        rhythmic_motifs = self._find_rhythmic_motifs(pitches, durations)
        motifs.extend(rhythmic_motifs)

        # Find opening/closing motifs (phrase structure)
        structural_motifs = self._find_structural_motifs(pitches, durations)
        motifs.extend(structural_motifs)

        # Add metadata
        for motif in motifs:
            for key, value in metadata.items():
                setattr(motif, key, value)

        # Remove duplicates
        motifs = self._deduplicate_motifs(motifs)

        return motifs

    def _find_repeated_subsequences(self, pitches: List[int],
                                   durations: List[float]) -> List[Motif]:
        """Find motifs that repeat within the sequence."""
        motifs = []
        subsequences = defaultdict(list)

        # Extract all subsequences
        for length in range(self.min_length, min(self.max_length, len(pitches)) + 1):
            for i in range(len(pitches) - length + 1):
                subseq_pitches = pitches[i:i+length]
                subseq_durations = durations[i:i+length]

                # Use intervals as key (transposition-invariant)
                intervals = tuple(subseq_pitches[j+1] - subseq_pitches[j]
                                for j in range(length - 1))

                subsequences[intervals].append((i, subseq_pitches, subseq_durations))

        # Find repeated subsequences
        for intervals, occurrences in subsequences.items():
            if len(occurrences) >= 2:  # Repeated at least twice
                # Use first occurrence
                pos, notes, rhythm = occurrences[0]

                contour = []
                for j in range(len(notes) - 1):
                    diff = notes[j+1] - notes[j]
                    contour.append(1 if diff > 0 else (-1 if diff < 0 else 0))

                motif = Motif(
                    id="",
                    notes=notes,
                    intervals=list(intervals),
                    rhythm=rhythm,
                    contour=contour,
                )
                motifs.append(motif)

        return motifs

    def _find_rhythmic_motifs(self, pitches: List[int],
                             durations: List[float]) -> List[Motif]:
        """Find motifs with distinctive rhythmic patterns."""
        motifs = []

        # Look for syncopation, dotted rhythms, etc.
        for i in range(len(pitches) - self.min_length + 1):
            for length in range(self.min_length, min(self.max_length, len(pitches) - i) + 1):
                rhythm_slice = durations[i:i+length]

                # Check for rhythmic interest
                if self._is_rhythmically_interesting(rhythm_slice):
                    pitch_slice = pitches[i:i+length]
                    intervals = [pitch_slice[j+1] - pitch_slice[j]
                               for j in range(length - 1)]
                    contour = [1 if intervals[j] > 0 else (-1 if intervals[j] < 0 else 0)
                             for j in range(len(intervals))]

                    motif = Motif(
                        id="",
                        notes=pitch_slice,
                        intervals=intervals,
                        rhythm=rhythm_slice,
                        contour=contour,
                    )
                    motifs.append(motif)

        return motifs

    def _find_structural_motifs(self, pitches: List[int],
                               durations: List[float]) -> List[Motif]:
        """Extract opening and closing motifs (phrase boundaries)."""
        motifs = []

        # Opening motif
        if len(pitches) >= self.min_length:
            length = min(self.max_length, len(pitches) // 4)  # First quarter
            if length >= self.min_length:
                pitch_slice = pitches[:length]
                rhythm_slice = durations[:length]

                intervals = [pitch_slice[j+1] - pitch_slice[j]
                           for j in range(length - 1)]
                contour = [1 if intervals[j] > 0 else (-1 if intervals[j] < 0 else 0)
                         for j in range(len(intervals))]

                opening_motif = Motif(
                    id="",
                    notes=pitch_slice,
                    intervals=intervals,
                    rhythm=rhythm_slice,
                    contour=contour,
                    emotion_tags=["opening"],
                )
                motifs.append(opening_motif)

        # Closing motif
        if len(pitches) >= self.min_length:
            length = min(self.max_length, len(pitches) // 4)
            if length >= self.min_length:
                pitch_slice = pitches[-length:]
                rhythm_slice = durations[-length:]

                intervals = [pitch_slice[j+1] - pitch_slice[j]
                           for j in range(length - 1)]
                contour = [1 if intervals[j] > 0 else (-1 if intervals[j] < 0 else 0)
                         for j in range(len(intervals))]

                closing_motif = Motif(
                    id="",
                    notes=pitch_slice,
                    intervals=intervals,
                    rhythm=rhythm_slice,
                    contour=contour,
                    emotion_tags=["closing"],
                )
                motifs.append(closing_motif)

        return motifs

    def _is_rhythmically_interesting(self, durations: List[float]) -> bool:
        """Check if rhythm has interesting patterns."""
        if len(set(durations)) == 1:  # All same duration
            return False

        # Check for dotted rhythms (3:1 ratio)
        for i in range(len(durations) - 1):
            ratio = durations[i] / durations[i+1] if durations[i+1] > 0 else 0
            if 2.5 <= ratio <= 3.5 or 1/3.5 <= ratio <= 1/2.5:
                return True

        return True  # Default to interesting if varied

    def _deduplicate_motifs(self, motifs: List[Motif]) -> List[Motif]:
        """Remove duplicate motifs based on interval content."""
        seen_intervals = set()
        unique_motifs = []

        for motif in motifs:
            interval_tuple = tuple(motif.intervals)
            if interval_tuple not in seen_intervals:
                seen_intervals.add(interval_tuple)
                unique_motifs.append(motif)

        return unique_motifs


class MotifDatabase:
    """
    Persistent database for storing and querying motifs.

    Supports tagging, similarity search, and statistical queries.
    """

    def __init__(self, db_path: str = "motif_library.json"):
        """
        Initialize motif database.

        Args:
            db_path: Path to JSON database file
        """
        self.db_path = db_path
        self.motifs: Dict[str, Motif] = {}
        self.tags_index: Dict[str, Set[str]] = defaultdict(set)  # tag -> motif_ids
        self.load()

    def add_motif(self, motif: Motif):
        """Add motif to database."""
        self.motifs[motif.id] = motif

        # Update tag indexes
        for tag in motif.emotion_tags:
            self.tags_index[f"emotion:{tag}"].add(motif.id)

        if motif.genre:
            self.tags_index[f"genre:{motif.genre}"].add(motif.id)

        if motif.composer:
            self.tags_index[f"composer:{motif.composer}"].add(motif.id)

        if motif.era:
            self.tags_index[f"era:{motif.era}"].add(motif.id)

        self.tags_index[f"contour:{motif.contour_type}"].add(motif.id)

    def search_by_tags(self, **tags) -> List[Motif]:
        """
        Search for motifs by tags.

        Args:
            **tags: emotion="joyful", genre="classical", etc.

        Returns:
            List of matching motifs
        """
        # Build query keys
        query_keys = []
        for tag_type, tag_value in tags.items():
            query_keys.append(f"{tag_type}:{tag_value}")

        # Find intersection of motif IDs
        result_ids = None
        for key in query_keys:
            motif_ids = self.tags_index.get(key, set())
            if result_ids is None:
                result_ids = motif_ids.copy()
            else:
                result_ids &= motif_ids

        if result_ids is None:
            return []

        return [self.motifs[mid] for mid in result_ids if mid in self.motifs]

    def find_similar(self, query_motif: Motif, top_k: int = 5,
                    metric: str = 'interval') -> List[Tuple[Motif, float]]:
        """
        Find similar motifs using various similarity metrics.

        Args:
            query_motif: Motif to search for
            top_k: Number of results to return
            metric: 'interval', 'contour', 'rhythm', or 'combined'

        Returns:
            List of (motif, similarity_score) tuples, sorted by similarity
        """
        similarities = []

        for motif_id, motif in self.motifs.items():
            if motif_id == query_motif.id:
                continue

            if metric == 'interval':
                sim = self._interval_similarity(query_motif, motif)
            elif metric == 'contour':
                sim = self._contour_similarity(query_motif, motif)
            elif metric == 'rhythm':
                sim = self._rhythm_similarity(query_motif, motif)
            else:  # combined
                sim = (
                    0.5 * self._interval_similarity(query_motif, motif) +
                    0.3 * self._contour_similarity(query_motif, motif) +
                    0.2 * self._rhythm_similarity(query_motif, motif)
                )

            similarities.append((motif, sim))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _interval_similarity(self, m1: Motif, m2: Motif) -> float:
        """Compute interval-based similarity using edit distance."""
        distance = self._edit_distance(m1.intervals, m2.intervals)
        max_len = max(len(m1.intervals), len(m2.intervals))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _contour_similarity(self, m1: Motif, m2: Motif) -> float:
        """Compute contour similarity using LCS."""
        lcs_len = self._lcs_length(m1.contour, m2.contour)
        max_len = max(len(m1.contour), len(m2.contour))
        return lcs_len / max_len if max_len > 0 else 0.0

    def _rhythm_similarity(self, m1: Motif, m2: Motif) -> float:
        """Compute rhythmic similarity."""
        distance = self._edit_distance(m1.rhythm, m2.rhythm)
        max_len = max(len(m1.rhythm), len(m2.rhythm))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def _edit_distance(self, seq1: List, seq2: List) -> int:
        """Levenshtein edit distance."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

        return dp[m][n]

    def _lcs_length(self, seq1: List, seq2: List) -> int:
        """Longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {
            'total_motifs': len(self.motifs),
            'avg_length': np.mean([m.length for m in self.motifs.values()]) if self.motifs else 0,
            'avg_range': np.mean([m.pitch_range for m in self.motifs.values()]) if self.motifs else 0,
            'composers': len(self.tags_index.get('composer', set())),
            'genres': len(self.tags_index.get('genre', set())),
            'emotions': len([k for k in self.tags_index.keys() if k.startswith('emotion:')]),
        }

        # Contour type distribution
        contour_dist = Counter()
        for motif in self.motifs.values():
            contour_dist[motif.contour_type] += 1
        stats['contour_distribution'] = dict(contour_dist)

        return stats

    def save(self):
        """Save database to JSON file."""
        data = {
            'motifs': {
                mid: asdict(motif) for mid, motif in self.motifs.items()
            }
        }

        with open(self.db_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load database from JSON file."""
        if not Path(self.db_path).exists():
            return

        with open(self.db_path, 'r') as f:
            data = json.load(f)

        for motif_data in data.get('motifs', {}).values():
            motif = Motif(**motif_data)
            self.add_motif(motif)


# Example usage
if __name__ == "__main__":
    print("Motif Library and Database System")
    print("=" * 60)

    # Create some example motifs
    motifs = []

    # "Ode to Joy" motif
    ode_to_joy = Motif(
        id="beethoven_ode",
        notes=[64, 64, 65, 67, 67, 65, 64, 62],
        intervals=[0, 1, 2, 0, -2, -1, -2],
        rhythm=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        contour=[0, 1, 1, 0, -1, -1, -1],
        source="Symphony No. 9",
        composer="Beethoven",
        genre="classical",
        era="romantic",
        emotion_tags=["joyful", "triumphant"],
    )
    motifs.append(ode_to_joy)

    # "Dies Irae" motif
    dies_irae = Motif(
        id="gregorian_dies",
        notes=[60, 62, 60, 58, 60, 57, 60],
        intervals=[2, -2, -2, 2, -3, 3],
        rhythm=[1.0, 1.0, 0.5, 0.5, 1.0, 1.0],
        contour=[1, -1, -1, 1, -1, 1],
        source="Dies Irae",
        composer="Gregorian Chant",
        genre="sacred",
        era="medieval",
        emotion_tags=["dark", "mysterious"],
    )
    motifs.append(dies_irae)

    # Initialize database
    db = MotifDatabase("test_motif_library.json")

    # Add motifs
    for motif in motifs:
        db.add_motif(motif)
        print(f"\nAdded motif: {motif.id}")
        print(f"  Source: {motif.source}")
        print(f"  Contour type: {motif.contour_type}")
        print(f"  Pitch range: {motif.pitch_range}")

    # Search by tags
    print("\n\nSearching for joyful motifs:")
    print("-" * 60)
    joyful_motifs = db.search_by_tags(emotion="joyful")
    for motif in joyful_motifs:
        print(f"  {motif.id}: {motif.source} ({motif.composer})")

    # Find similar motifs
    print("\n\nFinding motifs similar to 'Ode to Joy':")
    print("-" * 60)
    similar = db.find_similar(ode_to_joy, top_k=3, metric='combined')
    for motif, similarity in similar:
        print(f"  {motif.id}: similarity = {similarity:.3f}")

    # Transformations
    print("\n\nMotif Transformations:")
    print("-" * 60)
    print(f"Original: {ode_to_joy.notes}")
    print(f"Transposed (+2): {ode_to_joy.transpose(2).notes}")
    print(f"Retrograde: {ode_to_joy.retrograde().notes}")
    print(f"Inversion: {ode_to_joy.inversion().notes}")

    # Statistics
    print("\n\nDatabase Statistics:")
    print("-" * 60)
    stats = db.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Save database
    db.save()
    print(f"\n\nDatabase saved to: {db.db_path}")

    print("\n" + "=" * 60)
    print("Motif library complete!")
