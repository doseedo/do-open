#!/usr/bin/env python3
"""
Pattern Extractor for MIDI Analysis and Learning
==================================================

This module implements comprehensive pattern mining and extraction from MIDI files
using statistical methods, n-gram analysis, clustering, and motif discovery.

Features:
- Melodic pattern extraction (n-grams, contour patterns)
- Harmonic progression analysis
- Rhythmic motif discovery
- Clustering and pattern grouping
- Statistical analysis of musical features

Research foundations:
- Music Information Retrieval (Müller, 2015)
- Pattern Discovery in Music (Conklin & Witten, 1995)
- N-gram Models for Music (Pearce & Wiggins, 2012)
- Melodic Clustering (Typke et al., 2005)

Author: Agent 9 - ML Integration & Pattern Discovery
License: MIT
"""

from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
import pickle
from pathlib import Path
import itertools
import math
import heapq

try:
    import numpy as np
    from scipy import stats
    from scipy.spatial.distance import euclidean, cosine
    from scipy.cluster.hierarchy import linkage, fcluster
    from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Install with: pip install scikit-learn scipy")


@dataclass
class MelodicPattern:
    """Represents a discovered melodic pattern."""
    notes: List[int]  # MIDI pitch values
    intervals: List[int]  # Interval sequence
    contour: List[int]  # Melodic contour (-1, 0, 1)
    rhythm: List[float]  # Duration sequence
    frequency: int = 0  # Number of occurrences
    positions: List[int] = field(default_factory=list)  # Where it appears
    context: Dict[str, Any] = field(default_factory=dict)  # Additional metadata


@dataclass
class HarmonicPattern:
    """Represents a discovered harmonic progression."""
    chords: List[str]  # Chord symbols
    root_motion: List[int]  # Root note intervals
    chord_types: List[str]  # Chord quality (maj, min, etc.)
    frequency: int = 0
    positions: List[int] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RhythmicPattern:
    """Represents a discovered rhythmic motif."""
    durations: List[float]  # Note durations
    iois: List[float]  # Inter-onset intervals
    accent_pattern: List[float]  # Velocity/accent pattern
    metric_position: List[float]  # Position within measure
    frequency: int = 0
    positions: List[int] = field(default_factory=list)


class NGramExtractor:
    """
    N-gram pattern extraction for melodic, harmonic, and rhythmic sequences.

    Implements variable-order Markov models and statistical n-gram analysis.
    """

    def __init__(self, min_n: int = 2, max_n: int = 8):
        """
        Initialize n-gram extractor.

        Args:
            min_n: Minimum n-gram size
            max_n: Maximum n-gram size
        """
        self.min_n = min_n
        self.max_n = max_n
        self.pitch_ngrams: Dict[int, Counter] = {}
        self.interval_ngrams: Dict[int, Counter] = {}
        self.chord_ngrams: Dict[int, Counter] = {}
        self.rhythm_ngrams: Dict[int, Counter] = {}

    def extract_pitch_ngrams(self, pitches: List[int], n: int) -> List[Tuple[int, ...]]:
        """Extract n-grams from pitch sequence."""
        if len(pitches) < n:
            return []
        return [tuple(pitches[i:i+n]) for i in range(len(pitches) - n + 1)]

    def extract_interval_ngrams(self, pitches: List[int], n: int) -> List[Tuple[int, ...]]:
        """Extract n-grams from interval sequence."""
        intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches) - 1)]
        if len(intervals) < n:
            return []
        return [tuple(intervals[i:i+n]) for i in range(len(intervals) - n + 1)]

    def extract_contour_ngrams(self, pitches: List[int], n: int) -> List[Tuple[int, ...]]:
        """
        Extract melodic contour n-grams.

        Contour: -1 (down), 0 (same), 1 (up)
        """
        contour = []
        for i in range(len(pitches) - 1):
            diff = pitches[i+1] - pitches[i]
            if diff < 0:
                contour.append(-1)
            elif diff > 0:
                contour.append(1)
            else:
                contour.append(0)

        if len(contour) < n:
            return []
        return [tuple(contour[i:i+n]) for i in range(len(contour) - n + 1)]

    def extract_rhythm_ngrams(self, durations: List[float], n: int) -> List[Tuple[float, ...]]:
        """Extract n-grams from rhythmic duration sequence."""
        if len(durations) < n:
            return []
        # Quantize durations to common values
        quantized = [self._quantize_duration(d) for d in durations]
        return [tuple(quantized[i:i+n]) for i in range(len(quantized) - n + 1)]

    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest common rhythmic value."""
        common_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(common_durations, key=lambda x: abs(x - duration))

    def build_ngram_model(self, sequences: List[List[int]], ngram_type: str = 'pitch'):
        """
        Build n-gram frequency model from multiple sequences.

        Args:
            sequences: List of pitch/interval/chord sequences
            ngram_type: 'pitch', 'interval', 'contour', or 'rhythm'
        """
        for n in range(self.min_n, self.max_n + 1):
            ngrams = []
            for seq in sequences:
                if ngram_type == 'pitch':
                    ngrams.extend(self.extract_pitch_ngrams(seq, n))
                elif ngram_type == 'interval':
                    ngrams.extend(self.extract_interval_ngrams(seq, n))
                elif ngram_type == 'contour':
                    ngrams.extend(self.extract_contour_ngrams(seq, n))

            if ngram_type == 'pitch':
                self.pitch_ngrams[n] = Counter(ngrams)
            elif ngram_type == 'interval':
                self.interval_ngrams[n] = Counter(ngrams)
            elif ngram_type == 'contour':
                if not hasattr(self, 'contour_ngrams'):
                    self.contour_ngrams = {}
                self.contour_ngrams[n] = Counter(ngrams)

    def get_most_common_ngrams(self, n: int, ngram_type: str = 'pitch',
                              top_k: int = 10) -> List[Tuple[Tuple, int]]:
        """Get most frequent n-grams of size n."""
        if ngram_type == 'pitch' and n in self.pitch_ngrams:
            return self.pitch_ngrams[n].most_common(top_k)
        elif ngram_type == 'interval' and n in self.interval_ngrams:
            return self.interval_ngrams[n].most_common(top_k)
        elif ngram_type == 'contour' and hasattr(self, 'contour_ngrams') and n in self.contour_ngrams:
            return self.contour_ngrams[n].most_common(top_k)
        return []

    def compute_ngram_entropy(self, n: int, ngram_type: str = 'pitch') -> float:
        """
        Compute Shannon entropy of n-gram distribution.

        Higher entropy = more diverse/unpredictable patterns
        """
        if ngram_type == 'pitch' and n in self.pitch_ngrams:
            counter = self.pitch_ngrams[n]
        elif ngram_type == 'interval' and n in self.interval_ngrams:
            counter = self.interval_ngrams[n]
        else:
            return 0.0

        total = sum(counter.values())
        if total == 0:
            return 0.0

        entropy = 0.0
        for count in counter.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        return entropy

    def predict_next(self, context: Tuple[int, ...], ngram_type: str = 'pitch') -> List[Tuple[int, float]]:
        """
        Predict next element given context using n-gram model.

        Returns list of (element, probability) tuples.
        """
        n = len(context) + 1

        if ngram_type == 'pitch' and n in self.pitch_ngrams:
            counter = self.pitch_ngrams[n]
        elif ngram_type == 'interval' and n in self.interval_ngrams:
            counter = self.interval_ngrams[n]
        else:
            return []

        # Find all n-grams starting with context
        matches = {ngram[-1]: count for ngram, count in counter.items()
                  if ngram[:-1] == context}

        if not matches:
            return []

        total = sum(matches.values())
        return [(elem, count/total) for elem, count in
                sorted(matches.items(), key=lambda x: x[1], reverse=True)]


class MelodicClusterer:
    """
    Cluster melodic patterns using various similarity metrics.

    Implements multiple clustering algorithms and melodic similarity measures.
    """

    def __init__(self):
        """Initialize melodic clusterer."""
        self.patterns: List[MelodicPattern] = []
        self.clusters: Dict[int, List[int]] = {}
        self.cluster_centers: Dict[int, MelodicPattern] = {}

    def add_pattern(self, pattern: MelodicPattern):
        """Add a pattern to the collection."""
        self.patterns.append(pattern)

    def interval_similarity(self, p1: MelodicPattern, p2: MelodicPattern) -> float:
        """
        Compute interval-based similarity using edit distance.

        Returns similarity in [0, 1], where 1 is identical.
        """
        i1, i2 = p1.intervals, p2.intervals
        if not i1 or not i2:
            return 0.0

        # Levenshtein distance
        distance = self._edit_distance(i1, i2)
        max_len = max(len(i1), len(i2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def contour_similarity(self, p1: MelodicPattern, p2: MelodicPattern) -> float:
        """
        Compute contour-based similarity.

        Uses longest common subsequence of contour directions.
        """
        c1, c2 = p1.contour, p2.contour
        if not c1 or not c2:
            return 0.0

        lcs_len = self._lcs_length(c1, c2)
        max_len = max(len(c1), len(c2))
        return lcs_len / max_len if max_len > 0 else 0.0

    def rhythmic_similarity(self, p1: MelodicPattern, p2: MelodicPattern) -> float:
        """Compute rhythmic similarity based on duration patterns."""
        r1, r2 = p1.rhythm, p2.rhythm
        if not r1 or not r2:
            return 0.0

        # Quantize and compare
        q1 = [self._quantize_duration(d) for d in r1]
        q2 = [self._quantize_duration(d) for d in r2]

        distance = self._edit_distance(q1, q2)
        max_len = max(len(q1), len(q2))
        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def combined_similarity(self, p1: MelodicPattern, p2: MelodicPattern,
                           weights: Dict[str, float] = None) -> float:
        """
        Compute weighted combination of similarity metrics.

        Args:
            p1, p2: Patterns to compare
            weights: Dict with keys 'interval', 'contour', 'rhythm'
        """
        if weights is None:
            weights = {'interval': 0.5, 'contour': 0.3, 'rhythm': 0.2}

        sim = 0.0
        sim += weights.get('interval', 0.5) * self.interval_similarity(p1, p2)
        sim += weights.get('contour', 0.3) * self.contour_similarity(p1, p2)
        sim += weights.get('rhythm', 0.2) * self.rhythmic_similarity(p1, p2)

        return sim

    def _edit_distance(self, seq1: List, seq2: List) -> int:
        """Compute Levenshtein edit distance between sequences."""
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
        """Compute longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])

        return dp[m][n]

    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest common rhythmic value."""
        common_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(common_durations, key=lambda x: abs(x - duration))

    def cluster_hierarchical(self, n_clusters: int = 5,
                            similarity_metric: str = 'combined') -> Dict[int, List[int]]:
        """
        Perform hierarchical clustering of patterns.

        Args:
            n_clusters: Number of clusters
            similarity_metric: 'interval', 'contour', 'rhythm', or 'combined'

        Returns:
            Dictionary mapping cluster_id -> list of pattern indices
        """
        if not SKLEARN_AVAILABLE or len(self.patterns) < 2:
            return {0: list(range(len(self.patterns)))}

        # Build distance matrix
        n = len(self.patterns)
        distances = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                if similarity_metric == 'interval':
                    sim = self.interval_similarity(self.patterns[i], self.patterns[j])
                elif similarity_metric == 'contour':
                    sim = self.contour_similarity(self.patterns[i], self.patterns[j])
                elif similarity_metric == 'rhythm':
                    sim = self.rhythmic_similarity(self.patterns[i], self.patterns[j])
                else:
                    sim = self.combined_similarity(self.patterns[i], self.patterns[j])

                dist = 1.0 - sim  # Convert similarity to distance
                distances[i][j] = dist
                distances[j][i] = dist

        # Hierarchical clustering
        linkage_matrix = linkage(distances[np.triu_indices(n, k=1)], method='average')
        labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

        # Group patterns by cluster
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        self.clusters = dict(clusters)
        self._compute_cluster_centers()

        return self.clusters

    def cluster_kmeans(self, n_clusters: int = 5) -> Dict[int, List[int]]:
        """
        Perform k-means clustering on feature vectors.

        Converts patterns to numerical features for clustering.
        """
        if not SKLEARN_AVAILABLE or len(self.patterns) < n_clusters:
            return {0: list(range(len(self.patterns)))}

        # Extract features
        features = []
        for pattern in self.patterns:
            feat = self._pattern_to_features(pattern)
            features.append(feat)

        features = np.array(features)

        # Standardize
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_scaled)

        # Group patterns
        clusters = defaultdict(list)
        for idx, label in enumerate(labels):
            clusters[int(label)].append(idx)

        self.clusters = dict(clusters)
        self._compute_cluster_centers()

        return self.clusters

    def _pattern_to_features(self, pattern: MelodicPattern,
                            max_length: int = 20) -> np.ndarray:
        """
        Convert melodic pattern to fixed-length feature vector.

        Features include:
        - Interval statistics (mean, std, min, max)
        - Contour features
        - Rhythmic features
        - Pitch range
        """
        features = []

        # Interval statistics
        if pattern.intervals:
            intervals = np.array(pattern.intervals)
            features.extend([
                np.mean(intervals),
                np.std(intervals),
                np.min(intervals),
                np.max(intervals),
                len(intervals)
            ])
        else:
            features.extend([0, 0, 0, 0, 0])

        # Contour features
        if pattern.contour:
            contour = np.array(pattern.contour)
            features.extend([
                np.sum(contour == 1),   # Upward movements
                np.sum(contour == -1),  # Downward movements
                np.sum(contour == 0),   # Repeated notes
            ])
        else:
            features.extend([0, 0, 0])

        # Pitch range
        if pattern.notes:
            features.extend([
                max(pattern.notes) - min(pattern.notes),
                np.mean(pattern.notes),
            ])
        else:
            features.extend([0, 0])

        # Rhythm features
        if pattern.rhythm:
            rhythm = np.array(pattern.rhythm)
            features.extend([
                np.mean(rhythm),
                np.std(rhythm),
                np.sum(rhythm),
            ])
        else:
            features.extend([0, 0, 0])

        return np.array(features)

    def _compute_cluster_centers(self):
        """Compute representative pattern for each cluster (medoid)."""
        for cluster_id, pattern_indices in self.clusters.items():
            if not pattern_indices:
                continue

            # Find medoid (pattern with minimum average distance to others)
            min_avg_dist = float('inf')
            medoid_idx = pattern_indices[0]

            for i in pattern_indices:
                avg_dist = 0.0
                for j in pattern_indices:
                    if i != j:
                        sim = self.combined_similarity(self.patterns[i],
                                                       self.patterns[j])
                        avg_dist += (1.0 - sim)

                avg_dist /= len(pattern_indices)

                if avg_dist < min_avg_dist:
                    min_avg_dist = avg_dist
                    medoid_idx = i

            self.cluster_centers[cluster_id] = self.patterns[medoid_idx]

    def get_cluster_statistics(self) -> Dict[int, Dict[str, Any]]:
        """Get statistics for each cluster."""
        stats = {}

        for cluster_id, pattern_indices in self.clusters.items():
            patterns = [self.patterns[i] for i in pattern_indices]

            all_intervals = []
            all_lengths = []

            for p in patterns:
                all_intervals.extend(p.intervals)
                all_lengths.append(len(p.notes))

            stats[cluster_id] = {
                'size': len(patterns),
                'avg_length': np.mean(all_lengths) if all_lengths else 0,
                'avg_interval': np.mean(all_intervals) if all_intervals else 0,
                'center': self.cluster_centers.get(cluster_id),
            }

        return stats


class PatternExtractor:
    """
    Main pattern extraction engine for MIDI analysis.

    Combines n-gram analysis, clustering, and statistical methods to discover
    recurring patterns in melodic, harmonic, and rhythmic dimensions.
    """

    def __init__(self):
        """Initialize pattern extractor."""
        self.ngram_extractor = NGramExtractor()
        self.melodic_clusterer = MelodicClusterer()
        self.melodic_patterns: List[MelodicPattern] = []
        self.harmonic_patterns: List[HarmonicPattern] = []
        self.rhythmic_patterns: List[RhythmicPattern] = []

    def extract_melodic_patterns(self, pitch_sequences: List[List[int]],
                                 duration_sequences: List[List[float]] = None,
                                 min_length: int = 3,
                                 min_frequency: int = 2) -> List[MelodicPattern]:
        """
        Extract recurring melodic patterns from pitch sequences.

        Args:
            pitch_sequences: List of pitch sequences (MIDI note numbers)
            duration_sequences: Corresponding duration sequences
            min_length: Minimum pattern length
            min_frequency: Minimum number of occurrences

        Returns:
            List of discovered melodic patterns
        """
        patterns = []
        pattern_dict: Dict[Tuple, MelodicPattern] = {}

        for seq_idx, pitches in enumerate(pitch_sequences):
            durations = duration_sequences[seq_idx] if duration_sequences else None

            # Extract all subsequences
            for length in range(min_length, min(len(pitches), 20) + 1):
                for i in range(len(pitches) - length + 1):
                    subseq = pitches[i:i+length]

                    # Compute intervals and contour
                    intervals = tuple(subseq[j+1] - subseq[j]
                                    for j in range(len(subseq) - 1))

                    contour = []
                    for j in range(len(subseq) - 1):
                        diff = subseq[j+1] - subseq[j]
                        contour.append(1 if diff > 0 else (-1 if diff < 0 else 0))
                    contour = tuple(contour)

                    # Get rhythm if available
                    rhythm = durations[i:i+length] if durations else []

                    # Use interval sequence as key (transposition-invariant)
                    if intervals in pattern_dict:
                        pattern_dict[intervals].frequency += 1
                        pattern_dict[intervals].positions.append((seq_idx, i))
                    else:
                        pattern = MelodicPattern(
                            notes=subseq,
                            intervals=list(intervals),
                            contour=list(contour),
                            rhythm=rhythm,
                            frequency=1,
                            positions=[(seq_idx, i)]
                        )
                        pattern_dict[intervals] = pattern

        # Filter by frequency
        patterns = [p for p in pattern_dict.values() if p.frequency >= min_frequency]

        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        self.melodic_patterns = patterns
        return patterns

    def extract_harmonic_patterns(self, chord_sequences: List[List[str]],
                                  min_length: int = 2,
                                  min_frequency: int = 2) -> List[HarmonicPattern]:
        """
        Extract recurring harmonic progressions.

        Args:
            chord_sequences: List of chord symbol sequences
            min_length: Minimum progression length
            min_frequency: Minimum occurrences
        """
        pattern_dict: Dict[Tuple, HarmonicPattern] = {}

        for seq_idx, chords in enumerate(chord_sequences):
            for length in range(min_length, min(len(chords), 10) + 1):
                for i in range(len(chords) - length + 1):
                    subseq = chords[i:i+length]
                    chord_tuple = tuple(subseq)

                    if chord_tuple in pattern_dict:
                        pattern_dict[chord_tuple].frequency += 1
                        pattern_dict[chord_tuple].positions.append((seq_idx, i))
                    else:
                        pattern = HarmonicPattern(
                            chords=subseq,
                            root_motion=[],  # TODO: compute from chord roots
                            chord_types=[],  # TODO: extract chord qualities
                            frequency=1,
                            positions=[(seq_idx, i)]
                        )
                        pattern_dict[chord_tuple] = pattern

        patterns = [p for p in pattern_dict.values() if p.frequency >= min_frequency]
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        self.harmonic_patterns = patterns
        return patterns

    def extract_rhythmic_patterns(self, duration_sequences: List[List[float]],
                                  velocity_sequences: List[List[int]] = None,
                                  min_length: int = 2,
                                  min_frequency: int = 2) -> List[RhythmicPattern]:
        """
        Extract recurring rhythmic patterns.

        Args:
            duration_sequences: List of note duration sequences
            velocity_sequences: Optional velocity sequences
            min_length: Minimum pattern length
            min_frequency: Minimum occurrences
        """
        pattern_dict: Dict[Tuple, RhythmicPattern] = {}

        for seq_idx, durations in enumerate(duration_sequences):
            velocities = velocity_sequences[seq_idx] if velocity_sequences else None

            # Quantize durations
            quantized = [self._quantize_duration(d) for d in durations]

            for length in range(min_length, min(len(quantized), 16) + 1):
                for i in range(len(quantized) - length + 1):
                    subseq = tuple(quantized[i:i+length])

                    if subseq in pattern_dict:
                        pattern_dict[subseq].frequency += 1
                        pattern_dict[subseq].positions.append((seq_idx, i))
                    else:
                        accent = velocities[i:i+length] if velocities else []
                        pattern = RhythmicPattern(
                            durations=list(subseq),
                            iois=list(subseq),  # Simplified
                            accent_pattern=accent,
                            metric_position=[],
                            frequency=1,
                            positions=[(seq_idx, i)]
                        )
                        pattern_dict[subseq] = pattern

        patterns = [p for p in pattern_dict.values() if p.frequency >= min_frequency]
        patterns.sort(key=lambda p: p.frequency, reverse=True)

        self.rhythmic_patterns = patterns
        return patterns

    def _quantize_duration(self, duration: float) -> float:
        """Quantize duration to nearest common rhythmic value."""
        common_durations = [0.125, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        return min(common_durations, key=lambda x: abs(x - duration))

    def build_pattern_library(self, pitch_sequences: List[List[int]],
                             duration_sequences: List[List[float]] = None,
                             chord_sequences: List[List[str]] = None) -> Dict[str, Any]:
        """
        Build comprehensive pattern library from MIDI data.

        Returns:
            Dictionary containing all discovered patterns and statistics
        """
        library = {
            'melodic': [],
            'harmonic': [],
            'rhythmic': [],
            'statistics': {}
        }

        # Extract patterns
        library['melodic'] = self.extract_melodic_patterns(
            pitch_sequences, duration_sequences
        )

        if chord_sequences:
            library['harmonic'] = self.extract_harmonic_patterns(chord_sequences)

        if duration_sequences:
            library['rhythmic'] = self.extract_rhythmic_patterns(duration_sequences)

        # Build n-gram models
        self.ngram_extractor.build_ngram_model(pitch_sequences, 'pitch')
        self.ngram_extractor.build_ngram_model(pitch_sequences, 'interval')

        # Compute statistics
        library['statistics'] = {
            'num_melodic_patterns': len(library['melodic']),
            'num_harmonic_patterns': len(library['harmonic']),
            'num_rhythmic_patterns': len(library['rhythmic']),
            'total_sequences': len(pitch_sequences),
        }

        return library

    def save_patterns(self, filepath: str):
        """Save extracted patterns to file."""
        data = {
            'melodic_patterns': [self._pattern_to_dict(p) for p in self.melodic_patterns],
            'harmonic_patterns': [self._harmonic_to_dict(p) for p in self.harmonic_patterns],
            'rhythmic_patterns': [self._rhythmic_to_dict(p) for p in self.rhythmic_patterns],
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def _pattern_to_dict(self, pattern: MelodicPattern) -> Dict:
        """Convert MelodicPattern to dictionary."""
        return {
            'notes': pattern.notes,
            'intervals': pattern.intervals,
            'contour': pattern.contour,
            'rhythm': pattern.rhythm,
            'frequency': pattern.frequency,
            'positions': pattern.positions,
        }

    def _harmonic_to_dict(self, pattern: HarmonicPattern) -> Dict:
        """Convert HarmonicPattern to dictionary."""
        return {
            'chords': pattern.chords,
            'frequency': pattern.frequency,
            'positions': pattern.positions,
        }

    def _rhythmic_to_dict(self, pattern: RhythmicPattern) -> Dict:
        """Convert RhythmicPattern to dictionary."""
        return {
            'durations': pattern.durations,
            'frequency': pattern.frequency,
            'positions': pattern.positions,
        }


# Example usage
if __name__ == "__main__":
    print("Pattern Extractor for MIDI Analysis")
    print("=" * 50)

    # Example: Extract patterns from simple sequences
    pitch_sequences = [
        [60, 62, 64, 65, 67, 65, 64, 62],  # C major scale fragment
        [60, 62, 64, 65, 67, 69, 71, 72],  # C major scale
        [64, 65, 67, 65, 64, 62, 60, 62],  # Similar pattern
        [60, 62, 64, 65, 64, 62, 60, 59],  # Variation
    ]

    duration_sequences = [
        [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 1.0, 0.5, 0.5, 0.5, 0.5, 1.0],
        [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0],
    ]

    # Initialize extractor
    extractor = PatternExtractor()

    # Extract patterns
    print("\nExtracting melodic patterns...")
    melodic_patterns = extractor.extract_melodic_patterns(
        pitch_sequences, duration_sequences, min_frequency=2
    )

    print(f"\nFound {len(melodic_patterns)} melodic patterns:")
    for i, pattern in enumerate(melodic_patterns[:5], 1):
        print(f"\n{i}. Pattern (frequency: {pattern.frequency}):")
        print(f"   Notes: {pattern.notes}")
        print(f"   Intervals: {pattern.intervals}")
        print(f"   Contour: {pattern.contour}")

    # N-gram analysis
    print("\n\nN-gram Analysis:")
    print("-" * 50)
    extractor.ngram_extractor.build_ngram_model(pitch_sequences, 'interval')

    for n in range(2, 5):
        top_ngrams = extractor.ngram_extractor.get_most_common_ngrams(n, 'interval', top_k=3)
        print(f"\nMost common {n}-grams (intervals):")
        for ngram, count in top_ngrams:
            print(f"  {ngram}: {count} occurrences")

    # Clustering
    if SKLEARN_AVAILABLE and len(melodic_patterns) > 2:
        print("\n\nClustering Analysis:")
        print("-" * 50)

        for pattern in melodic_patterns:
            extractor.melodic_clusterer.add_pattern(pattern)

        clusters = extractor.melodic_clusterer.cluster_hierarchical(n_clusters=2)
        stats = extractor.melodic_clusterer.get_cluster_statistics()

        for cluster_id, cluster_stats in stats.items():
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {cluster_stats['size']} patterns")
            print(f"  Avg length: {cluster_stats['avg_length']:.1f} notes")

    print("\n" + "=" * 50)
    print("Pattern extraction complete!")
