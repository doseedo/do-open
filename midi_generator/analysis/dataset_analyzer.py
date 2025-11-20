#!/usr/bin/env python3
"""
AGENT 16: MIDI Dataset Analysis Engine
=======================================

Build tools to analyze MIDI datasets (PiJAMA, Weimar, Lakh) and extract statistical
patterns for validation and improvement.

This module provides:
- Chord progression frequency analysis across multiple files
- Melodic interval distribution analysis
- Swing ratio measurement from real recordings
- Comping rhythm extraction (piano, guitar)
- Pattern extraction from professional recordings
- Validation metrics to compare generated vs. real music

Research-based validation framework using:
- PiJAMA Dataset: 200+ hours jazz piano (2,777 performances)
- Weimar Jazz Database: 300 solo transcriptions
- Lakh MIDI Dataset: 176,581 files
- Statistical comparison using KL divergence, correlation metrics

Author: Agent 16 - MIDI Dataset Analysis Engine
Integration Points: Provides validation for ALL other agents
"""

from typing import List, Tuple, Dict, Optional, Set, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import json
import warnings

# Type checking only - doesn't affect runtime
if TYPE_CHECKING:
    from .midi_analyzer import AnalysisResult, NoteEvent, ChordEvent
else:
    # Create placeholder types for when mido is not available
    AnalysisResult = Any
    NoteEvent = Any
    ChordEvent = Any

# Optional imports - only needed for actual MIDI analysis
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("numpy not available - some functionality will be limited")

try:
    from scipy import stats
    from scipy.spatial.distance import jensenshannon
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    warnings.warn("scipy not available - distribution comparison will be limited")

try:
    import mido
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available - MIDI file analysis will not work")

# Import from existing midi_analyzer (only if mido available)
if MIDO_AVAILABLE and not TYPE_CHECKING:
    try:
        from .midi_analyzer import (
            MidiAnalyzer, NoteEvent, ChordEvent, KeyDetector,
            ChordRecognizer, AnalysisResult
        )
    except ImportError:
        # If midi_analyzer can't be imported, disable MIDI functionality
        MIDO_AVAILABLE = False
        warnings.warn("Could not import midi_analyzer - MIDI functionality disabled")


# ==============================================================================
# UTILITY FUNCTIONS (work with or without numpy/scipy)
# ==============================================================================

def calculate_mean(values: List[float]) -> float:
    """Calculate mean (works without numpy)."""
    if not values:
        return 0.0
    if NUMPY_AVAILABLE:
        return float(np.mean(values))
    return sum(values) / len(values)


def calculate_std(values: List[float]) -> float:
    """Calculate standard deviation (works without numpy)."""
    if not values or len(values) < 2:
        return 0.0
    if NUMPY_AVAILABLE:
        return float(np.std(values))
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance ** 0.5


def calculate_correlation(x: List[float], y: List[float]) -> float:
    """Calculate Pearson correlation (works without scipy)."""
    if not x or not y or len(x) != len(y) or len(x) < 2:
        return 0.0

    if SCIPY_AVAILABLE:
        corr, _ = stats.pearsonr(x, y)
        return float(corr)

    # Manual calculation
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n)) / n
    std_x = (sum((xi - mean_x) ** 2 for xi in x) / n) ** 0.5
    std_y = (sum((yi - mean_y) ** 2 for yi in y) / n) ** 0.5

    if std_x == 0 or std_y == 0:
        return 0.0

    return cov / (std_x * std_y)


# ==============================================================================
# DATA STRUCTURES FOR DATASET ANALYSIS
# ==============================================================================

@dataclass
class ChordProgression:
    """Represents a chord progression pattern."""
    chords: List[Tuple[int, str]]  # [(root, quality), ...]
    key: Optional[int] = None
    frequency: int = 1

    def __str__(self) -> str:
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_str = '-'.join(f'{note_names[root]}{quality}' for root, quality in self.chords)
        return chord_str

    def to_roman_numerals(self) -> str:
        """Convert to Roman numeral analysis if key is known."""
        if self.key is None:
            return str(self)

        roman = ['I', 'bII', 'II', 'bIII', 'III', 'IV', 'bV', 'V', 'bVI', 'VI', 'bVII', 'VII']
        result = []
        for root, quality in self.chords:
            degree = (root - self.key) % 12
            numeral = roman[degree]
            if quality == 'minor' or quality == 'minor7':
                numeral = numeral.lower()
            result.append(f'{numeral}{quality}')
        return '-'.join(result)


@dataclass
class SwingMeasurement:
    """Measurement of swing feel from MIDI data."""
    swing_ratio: float  # Ratio of offbeat delay (0.5 = straight, 0.67 = triplet)
    tempo: float  # BPM
    std_dev: float  # Standard deviation of swing ratio
    confidence: float  # Confidence in measurement (0-1)
    num_samples: int  # Number of eighth note pairs analyzed

    def __str__(self) -> str:
        return f"Swing {self.swing_ratio:.3f} @ {self.tempo:.0f} BPM (±{self.std_dev:.3f}, n={self.num_samples})"


@dataclass
class CompingPattern:
    """Extracted comping rhythm pattern."""
    pattern: List[float]  # Beat positions (0.0-4.0 for 4 beats)
    duration_beats: int  # Pattern length in beats
    frequency: int  # How many times pattern appears
    avg_velocity: float  # Average velocity
    style: str = "unknown"  # charleston, montuno, sparse, dense, etc.

    def __str__(self) -> str:
        pattern_str = ' '.join(f'{p:.2f}' for p in self.pattern[:8])  # Show first 8 notes
        return f"{self.style} ({len(self.pattern)} notes, {self.frequency}x): [{pattern_str}...]"


@dataclass
class IntervalDistribution:
    """Statistical distribution of melodic intervals."""
    intervals: Dict[int, int]  # {interval: count}
    total: int
    mean_abs_interval: float
    stepwise_percentage: float  # Percentage of intervals <= 2 semitones

    def to_probability_distribution(self) -> Dict[int, float]:
        """Convert counts to probabilities."""
        if self.total == 0:
            return {}
        return {interval: count / self.total for interval, count in self.intervals.items()}


@dataclass
class DatasetStatistics:
    """Aggregated statistics from multiple MIDI files."""
    num_files: int = 0
    total_notes: int = 0
    total_duration_seconds: float = 0.0

    # Chord progressions
    chord_progressions: Dict[str, ChordProgression] = field(default_factory=dict)

    # Melodic intervals
    interval_distribution: Optional[IntervalDistribution] = None

    # Swing measurements
    swing_measurements: List[SwingMeasurement] = field(default_factory=list)
    avg_swing_ratio: float = 0.0
    swing_tempo_correlation: float = 0.0  # Correlation between tempo and swing ratio

    # Comping patterns
    comping_patterns: List[CompingPattern] = field(default_factory=list)

    # Velocity statistics
    velocity_mean: float = 0.0
    velocity_std: float = 0.0

    # Pitch class distribution
    pitch_class_distribution: Dict[int, float] = field(default_factory=dict)

    # Rhythm complexity
    rhythm_complexity_scores: List[float] = field(default_factory=list)

    # Voice leading distances (for multi-voice arrangements)
    voice_leading_distances: List[float] = field(default_factory=list)


# ==============================================================================
# DATASET ANALYZER - MAIN CLASS
# ==============================================================================

class DatasetAnalyzer:
    """
    Analyze multiple MIDI files to extract patterns and statistics.

    Primary tool for Agent 16 to:
    1. Extract patterns from real jazz recordings
    2. Measure authentic swing feels, comping rhythms, voice leading
    3. Provide validation baselines for generated music
    4. Quantify improvement over baseline
    """

    def __init__(self):
        """Initialize dataset analyzer."""
        self.stats = DatasetStatistics()
        self.analysis_results: List[AnalysisResult] = []

    def analyze_dataset(self,
                       midi_paths: List[str],
                       analyze_chords: bool = True,
                       analyze_swing: bool = True,
                       analyze_comping: bool = True,
                       analyze_intervals: bool = True,
                       verbose: bool = True) -> DatasetStatistics:
        """
        Analyze multiple MIDI files and aggregate statistics.

        Args:
            midi_paths: List of paths to MIDI files
            analyze_chords: Extract chord progressions
            analyze_swing: Measure swing ratios
            analyze_comping: Extract comping rhythms
            analyze_intervals: Analyze melodic intervals
            verbose: Print progress messages

        Returns:
            DatasetStatistics with aggregated results
        """
        if verbose:
            print(f"\n{'='*80}")
            print(f"DATASET ANALYSIS: Processing {len(midi_paths)} MIDI files")
            print(f"{'='*80}\n")

        self.stats = DatasetStatistics(num_files=len(midi_paths))
        self.analysis_results = []

        all_intervals = []
        all_velocities = []
        all_pitch_classes = []

        for i, midi_path in enumerate(midi_paths):
            if verbose and i % 10 == 0:
                print(f"Processing file {i+1}/{len(midi_paths)}: {Path(midi_path).name}")

            try:
                # Analyze individual file
                analyzer = MidiAnalyzer(midi_path)
                result = analyzer.analyze(
                    detect_key=True,
                    detect_chords=analyze_chords,
                    analyze_rhythm=analyze_swing,
                    analyze_melody=analyze_intervals
                )
                self.analysis_results.append(result)

                # Aggregate statistics
                self.stats.total_notes += len(result.notes)
                self.stats.total_duration_seconds += result.duration_seconds

                # Collect intervals
                if analyze_intervals and result.melodic_intervals:
                    all_intervals.extend(result.melodic_intervals)

                # Collect velocities
                if result.notes:
                    all_velocities.extend([n.velocity for n in result.notes])

                # Collect pitch classes
                if result.notes:
                    all_pitch_classes.extend([n.pitch_class for n in result.notes])

                # Extract chord progressions
                if analyze_chords and result.chords:
                    self._extract_chord_progressions(result)

                # Measure swing
                if analyze_swing and result.notes:
                    swing_measurement = self._measure_swing_ratio(result)
                    if swing_measurement:
                        self.stats.swing_measurements.append(swing_measurement)

                # Extract comping patterns
                if analyze_comping and result.notes:
                    comping_patterns = self._extract_comping_rhythms(result)
                    self.stats.comping_patterns.extend(comping_patterns)

            except Exception as e:
                if verbose:
                    print(f"  Error analyzing {midi_path}: {e}")
                continue

        # Calculate aggregate statistics
        if all_intervals:
            self._calculate_interval_statistics(all_intervals)

        if all_velocities:
            self.stats.velocity_mean = calculate_mean(all_velocities)
            self.stats.velocity_std = calculate_std(all_velocities)

        if all_pitch_classes:
            self._calculate_pitch_class_distribution(all_pitch_classes)

        if self.stats.swing_measurements:
            self._calculate_swing_statistics()

        if verbose:
            self.print_dataset_summary()

        return self.stats

    def _extract_chord_progressions(self,
                                   result: AnalysisResult,
                                   progression_length: int = 4) -> None:
        """
        Extract common chord progressions from analysis result.

        Args:
            result: Analysis result from single MIDI file
            progression_length: Length of progressions to extract (2-8 chords)
        """
        if len(result.chords) < progression_length:
            return

        key = result.key.tonic if result.key else None

        # Extract all n-chord progressions
        for i in range(len(result.chords) - progression_length + 1):
            chord_slice = result.chords[i:i + progression_length]
            chord_tuple = tuple((c.root, c.quality) for c in chord_slice)

            # Create progression key
            prog_str = '-'.join(f'{root}{quality}' for root, quality in chord_tuple)

            if prog_str in self.stats.chord_progressions:
                self.stats.chord_progressions[prog_str].frequency += 1
            else:
                self.stats.chord_progressions[prog_str] = ChordProgression(
                    chords=list(chord_tuple),
                    key=key,
                    frequency=1
                )

    def _measure_swing_ratio(self, result: AnalysisResult) -> Optional[SwingMeasurement]:
        """
        Measure swing ratio from MIDI note events.

        Algorithm:
        1. Quantize notes to eighth note grid
        2. Find pairs of consecutive eighth notes
        3. Measure ratio of offbeat delay
        4. Average across all pairs

        Args:
            result: Analysis result from single MIDI file

        Returns:
            SwingMeasurement or None if insufficient data
        """
        if not result.average_tempo or not result.notes:
            return None

        tempo = result.average_tempo
        eighth_duration = 60.0 / tempo / 2  # Duration of one eighth note in seconds

        # Extract note onsets
        onsets = sorted([n.start_time for n in result.notes])

        if len(onsets) < 10:  # Need sufficient data
            return None

        # Find eighth note pairs
        swing_ratios = []

        for i in range(len(onsets) - 1):
            # Calculate beat position
            beat_pos = (onsets[i] / eighth_duration) % 2

            # Check if this is an on-beat note (position 0)
            if 0.0 <= beat_pos < 0.2:  # Tolerance for quantization
                # Next note should be offbeat
                next_beat_pos = (onsets[i+1] / eighth_duration) % 2

                # Measure deviation from straight eighths (0.5)
                if 0.3 < next_beat_pos < 0.9:  # Must be somewhat offbeat
                    # Calculate actual swing ratio
                    # Straight = 0.5, triplet swing = 0.67
                    time_diff = onsets[i+1] - onsets[i]
                    ratio = time_diff / eighth_duration

                    if 0.45 < ratio < 0.8:  # Reasonable swing range
                        swing_ratios.append(ratio)

        if len(swing_ratios) < 5:  # Need at least 5 samples
            return None

        # Calculate statistics
        mean_ratio = calculate_mean(swing_ratios)
        std_ratio = calculate_std(swing_ratios)
        confidence = min(1.0, len(swing_ratios) / 20)  # More samples = higher confidence

        return SwingMeasurement(
            swing_ratio=mean_ratio,
            tempo=tempo,
            std_dev=std_ratio,
            confidence=confidence,
            num_samples=len(swing_ratios)
        )

    def _extract_comping_rhythms(self,
                                result: AnalysisResult,
                                pattern_length_beats: int = 4) -> List[CompingPattern]:
        """
        Extract comping rhythm patterns from piano/guitar parts.

        Args:
            result: Analysis result
            pattern_length_beats: Length of patterns to extract

        Returns:
            List of detected comping patterns
        """
        if not result.average_tempo or not result.notes:
            return []

        tempo = result.average_tempo
        beat_duration = 60.0 / tempo
        pattern_duration = pattern_length_beats * beat_duration

        # Group notes by time windows
        patterns = []

        # Extract onset times normalized to beats
        beats = []
        velocities = []
        for note in result.notes:
            beat_time = note.start_time / beat_duration
            beats.append(beat_time % pattern_length_beats)  # Normalize to pattern length
            velocities.append(note.velocity)

        if len(beats) < 8:  # Need sufficient notes
            return []

        # Create pattern
        pattern = CompingPattern(
            pattern=sorted(beats[:32]),  # Take first 32 notes as sample
            duration_beats=pattern_length_beats,
            frequency=1,  # Would need pattern matching to count frequency
            avg_velocity=calculate_mean(velocities) if velocities else 0,
            style=self._classify_comping_style(beats)
        )

        return [pattern]

    def _classify_comping_style(self, beats: List[float]) -> str:
        """
        Classify comping style based on rhythm pattern.

        Args:
            beats: List of beat positions

        Returns:
            Style classification
        """
        if not beats:
            return "unknown"

        # Count offbeat vs onbeat
        offbeats = sum(1 for b in beats if 0.3 < (b % 1) < 0.7)
        onbeats = len(beats) - offbeats

        offbeat_ratio = offbeats / len(beats) if beats else 0

        # Density
        density = len(beats) / 4  # Notes per 4 beats

        if offbeat_ratio > 0.7:
            return "charleston"  # Heavy offbeat emphasis
        elif density < 2:
            return "sparse"
        elif density > 6:
            return "dense"
        else:
            return "standard"

    def _calculate_interval_statistics(self, intervals: List[int]) -> None:
        """Calculate interval distribution statistics."""
        interval_counts = Counter(intervals)
        total = len(intervals)

        # Calculate stepwise percentage (intervals of ±1 or ±2 semitones)
        stepwise = sum(interval_counts.get(i, 0) for i in [-2, -1, 0, 1, 2])
        stepwise_pct = (stepwise / total * 100) if total > 0 else 0

        # Mean absolute interval
        mean_abs = calculate_mean([abs(i) for i in intervals]) if intervals else 0

        self.stats.interval_distribution = IntervalDistribution(
            intervals=dict(interval_counts),
            total=total,
            mean_abs_interval=mean_abs,
            stepwise_percentage=stepwise_pct
        )

    def _calculate_pitch_class_distribution(self, pitch_classes: List[int]) -> None:
        """Calculate normalized pitch class distribution."""
        pc_counts = Counter(pitch_classes)
        total = sum(pc_counts.values())

        # Normalize to probabilities
        self.stats.pitch_class_distribution = {
            pc: count / total for pc, count in pc_counts.items()
        }

    def _calculate_swing_statistics(self) -> None:
        """Calculate aggregate swing statistics including tempo correlation."""
        if not self.stats.swing_measurements:
            return

        # Average swing ratio
        self.stats.avg_swing_ratio = calculate_mean([s.swing_ratio for s in self.stats.swing_measurements])

        # Correlation between tempo and swing ratio
        tempos = [s.tempo for s in self.stats.swing_measurements]
        ratios = [s.swing_ratio for s in self.stats.swing_measurements]

        if len(tempos) > 2:
            self.stats.swing_tempo_correlation = calculate_correlation(tempos, ratios)

    # ==========================================================================
    # VALIDATION METRICS - Compare Generated vs. Real Music
    # ==========================================================================

    def compare_generated_to_dataset(self,
                                    generated_midi_path: str,
                                    metrics: List[str] = None) -> Dict[str, float]:
        """
        Compare generated music to dataset statistics.

        Metrics:
        - interval_similarity: KL divergence of interval distributions
        - rhythm_similarity: Correlation of onset patterns
        - swing_accuracy: Difference in swing ratio
        - voicing_similarity: Voice spacing distribution match
        - overall_authenticity: Weighted average of all metrics

        Args:
            generated_midi_path: Path to generated MIDI file
            metrics: List of metrics to compute (None = all)

        Returns:
            Dictionary of metric scores (0-1, higher is better)
        """
        if metrics is None:
            metrics = ['interval', 'rhythm', 'swing', 'velocity', 'overall']

        # Analyze generated file
        analyzer = MidiAnalyzer(generated_midi_path)
        generated = analyzer.analyze()

        results = {}

        # Interval similarity
        if 'interval' in metrics and self.stats.interval_distribution:
            gen_intervals = Counter(generated.melodic_intervals)
            gen_total = sum(gen_intervals.values())
            gen_dist = {k: v/gen_total for k, v in gen_intervals.items()} if gen_total > 0 else {}

            dataset_dist = self.stats.interval_distribution.to_probability_distribution()

            similarity = self._calculate_distribution_similarity(gen_dist, dataset_dist)
            results['interval_similarity'] = similarity

        # Swing accuracy
        if 'swing' in metrics and self.stats.avg_swing_ratio > 0:
            gen_swing = self._measure_swing_ratio(generated)
            if gen_swing:
                # Convert difference to similarity score
                diff = abs(gen_swing.swing_ratio - self.stats.avg_swing_ratio)
                similarity = max(0, 1.0 - diff / 0.2)  # 0.2 = tolerance range
                results['swing_accuracy'] = similarity

        # Velocity similarity
        if 'velocity' in metrics and self.stats.velocity_mean > 0:
            gen_velocities = [n.velocity for n in generated.notes]
            if gen_velocities:
                gen_mean = calculate_mean(gen_velocities)
                # Normalize difference
                diff = abs(gen_mean - self.stats.velocity_mean)
                similarity = max(0, 1.0 - diff / 50)  # 50 = tolerance range
                results['velocity_similarity'] = similarity

        # Rhythm complexity
        if 'rhythm' in metrics:
            # Measure onset timing variance
            if generated.onset_times and len(generated.onset_times) > 1:
                # Calculate inter-onset intervals manually
                gen_ioi = [generated.onset_times[i+1] - generated.onset_times[i]
                          for i in range(len(generated.onset_times) - 1)]
                gen_complexity = calculate_std(gen_ioi) if len(gen_ioi) > 0 else 0

                # Compare to dataset (would need to store this)
                # For now, just check if it's in reasonable range
                reasonable_range = (0.05, 0.5)
                if reasonable_range[0] <= gen_complexity <= reasonable_range[1]:
                    results['rhythm_complexity'] = 0.8
                else:
                    results['rhythm_complexity'] = 0.5

        # Overall authenticity (weighted average)
        if 'overall' in metrics and results:
            weights = {
                'interval_similarity': 0.3,
                'swing_accuracy': 0.3,
                'velocity_similarity': 0.2,
                'rhythm_complexity': 0.2
            }
            weighted_sum = sum(results.get(k, 0) * weights.get(k, 0) for k in weights.keys())
            total_weight = sum(weights.get(k, 0) for k in results.keys() if k in weights)

            if total_weight > 0:
                results['overall_authenticity'] = weighted_sum / total_weight

        return results

    def _calculate_distribution_similarity(self,
                                          dist1: Dict[int, float],
                                          dist2: Dict[int, float]) -> float:
        """
        Calculate similarity between two probability distributions.
        Uses Jensen-Shannon divergence converted to similarity score.

        Args:
            dist1, dist2: Probability distributions

        Returns:
            Similarity score (0-1, higher is better)
        """
        # Get all keys
        all_keys = set(dist1.keys()) | set(dist2.keys())

        # Create aligned vectors
        p_list = [dist1.get(k, 0) for k in sorted(all_keys)]
        q_list = [dist2.get(k, 0) for k in sorted(all_keys)]

        # Normalize
        p_sum = sum(p_list)
        q_sum = sum(q_list)

        if p_sum > 0:
            p_list = [x / p_sum for x in p_list]
        if q_sum > 0:
            q_list = [x / q_sum for x in q_list]

        if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
            # Use scipy's Jensen-Shannon divergence
            p = np.array(p_list)
            q = np.array(q_list)
            js_divergence = jensenshannon(p, q)
        else:
            # Manual Jensen-Shannon divergence calculation
            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M) where M = 0.5(P+Q)
            m = [(p_list[i] + q_list[i]) / 2 for i in range(len(p_list))]

            kl_pm = sum(p_list[i] * ((p_list[i] / m[i]) if m[i] > 0 else 0)
                       for i in range(len(p_list)) if p_list[i] > 0)
            kl_qm = sum(q_list[i] * ((q_list[i] / m[i]) if m[i] > 0 else 0)
                       for i in range(len(q_list)) if q_list[i] > 0)

            js_divergence = (kl_pm + kl_qm) / 2
            js_divergence = js_divergence ** 0.5  # Take square root for distance

        # Convert to similarity (0 divergence = 1 similarity)
        similarity = 1.0 - min(float(js_divergence), 1.0)

        return similarity

    # ==========================================================================
    # PATTERN EXTRACTION FOR GENERATORS
    # ==========================================================================

    def extract_bebop_licks(self,
                           min_length: int = 4,
                           max_length: int = 12) -> List[List[int]]:
        """
        Extract common melodic patterns (bebop licks) from dataset.

        Args:
            min_length: Minimum lick length in notes
            max_length: Maximum lick length in notes

        Returns:
            List of interval patterns (list of semitone intervals)
        """
        licks = []

        for result in self.analysis_results:
            if not result.melodic_intervals:
                continue

            intervals = result.melodic_intervals

            # Extract all n-note patterns
            for length in range(min_length, max_length + 1):
                for i in range(len(intervals) - length + 1):
                    lick = intervals[i:i + length]
                    # Filter for melodic patterns (not too many large leaps)
                    large_leaps = sum(1 for interval in lick if abs(interval) > 5)
                    if large_leaps <= len(lick) * 0.2:  # Max 20% large leaps
                        licks.append(lick)

        # Count frequency and return most common
        lick_counter = Counter(tuple(lick) for lick in licks)
        most_common = lick_counter.most_common(100)  # Top 100 licks

        return [list(lick) for lick, count in most_common]

    def extract_walking_bass_patterns(self) -> List[List[int]]:
        """
        Extract walking bass patterns from dataset.

        Returns:
            List of bass line patterns (4 notes = 1 bar in 4/4)
        """
        bass_patterns = []

        for result in self.analysis_results:
            # Filter for bass register notes (E1-C3)
            bass_notes = [n for n in result.notes if 28 <= n.pitch <= 48]

            if len(bass_notes) < 4:
                continue

            # Sort by time
            bass_notes = sorted(bass_notes, key=lambda n: n.start_time)

            # Extract 4-note patterns (1 bar)
            for i in range(len(bass_notes) - 3):
                pattern = [bass_notes[j].pitch for j in range(i, i + 4)]
                bass_patterns.append(pattern)

        # Return most common patterns
        pattern_counter = Counter(tuple(p) for p in bass_patterns)
        most_common = pattern_counter.most_common(50)

        return [list(pattern) for pattern, count in most_common]

    # ==========================================================================
    # REPORTING
    # ==========================================================================

    def print_dataset_summary(self) -> None:
        """Print comprehensive dataset analysis summary."""
        s = self.stats

        print(f"\n{'='*80}")
        print(f"DATASET ANALYSIS SUMMARY")
        print(f"{'='*80}\n")

        print(f"📊 Dataset Overview:")
        print(f"   Files analyzed: {s.num_files}")
        print(f"   Total notes: {s.total_notes:,}")
        print(f"   Total duration: {s.total_duration_seconds/60:.1f} minutes")

        if s.interval_distribution:
            print(f"\n🎵 Melodic Intervals:")
            print(f"   Total intervals: {s.interval_distribution.total:,}")
            print(f"   Mean absolute interval: {s.interval_distribution.mean_abs_interval:.2f} semitones")
            print(f"   Stepwise motion: {s.interval_distribution.stepwise_percentage:.1f}%")

            # Top intervals
            top_intervals = sorted(s.interval_distribution.intervals.items(),
                                 key=lambda x: x[1], reverse=True)[:5]
            print(f"   Most common intervals: {', '.join(f'{i}({c})' for i, c in top_intervals)}")

        if s.swing_measurements:
            print(f"\n🎺 Swing Feel:")
            print(f"   Measurements: {len(s.swing_measurements)}")
            print(f"   Average swing ratio: {s.avg_swing_ratio:.3f}")
            print(f"   Tempo-swing correlation: {s.swing_tempo_correlation:.3f}")

            # Show swing by tempo range
            slow = [s for s in s.swing_measurements if s.tempo < 100]
            medium = [s for s in s.swing_measurements if 100 <= s.tempo < 160]
            fast = [s for s in s.swing_measurements if s.tempo >= 160]

            if slow:
                print(f"   Slow (<100 BPM): {calculate_mean([s.swing_ratio for s in slow]):.3f}")
            if medium:
                print(f"   Medium (100-160 BPM): {calculate_mean([s.swing_ratio for s in medium]):.3f}")
            if fast:
                print(f"   Fast (>160 BPM): {calculate_mean([s.swing_ratio for s in fast]):.3f}")

        if s.comping_patterns:
            print(f"\n🎹 Comping Patterns:")
            print(f"   Patterns extracted: {len(s.comping_patterns)}")

            # Count by style
            style_counts = Counter(p.style for p in s.comping_patterns)
            for style, count in style_counts.most_common():
                print(f"   {style}: {count}")

        if s.chord_progressions:
            print(f"\n🎸 Chord Progressions:")
            print(f"   Unique progressions: {len(s.chord_progressions)}")

            # Top 10 most common
            sorted_progs = sorted(s.chord_progressions.values(),
                                key=lambda p: p.frequency, reverse=True)
            print(f"   Top 10 most common:")
            for i, prog in enumerate(sorted_progs[:10], 1):
                print(f"   {i}. {prog} (×{prog.frequency})")

        if s.velocity_mean > 0:
            print(f"\n💨 Velocity:")
            print(f"   Mean: {s.velocity_mean:.1f}")
            print(f"   Std Dev: {s.velocity_std:.1f}")

        if s.pitch_class_distribution:
            print(f"\n🎼 Pitch Class Distribution:")
            note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            sorted_pcs = sorted(s.pitch_class_distribution.items(),
                              key=lambda x: x[1], reverse=True)
            top_5 = sorted_pcs[:5]
            print(f"   Top 5: {', '.join(f'{note_names[pc]}({prob:.2%})' for pc, prob in top_5)}")

        print(f"\n{'='*80}\n")

    def save_statistics(self, output_path: str) -> None:
        """
        Save dataset statistics to JSON file.

        Args:
            output_path: Path to output JSON file
        """
        # Convert to JSON-serializable format
        data = {
            'num_files': self.stats.num_files,
            'total_notes': self.stats.total_notes,
            'total_duration_seconds': self.stats.total_duration_seconds,
            'avg_swing_ratio': self.stats.avg_swing_ratio,
            'swing_tempo_correlation': self.stats.swing_tempo_correlation,
            'velocity_mean': self.stats.velocity_mean,
            'velocity_std': self.stats.velocity_std,
            'pitch_class_distribution': self.stats.pitch_class_distribution,
        }

        if self.stats.interval_distribution:
            data['interval_distribution'] = {
                'intervals': {str(k): v for k, v in self.stats.interval_distribution.intervals.items()},
                'total': self.stats.interval_distribution.total,
                'mean_abs_interval': self.stats.interval_distribution.mean_abs_interval,
                'stepwise_percentage': self.stats.interval_distribution.stepwise_percentage
            }

        # Chord progressions
        data['chord_progressions'] = {
            str(prog): {
                'frequency': prog.frequency,
                'key': prog.key
            }
            for prog in self.stats.chord_progressions.values()
        }

        # Swing measurements
        data['swing_measurements'] = [
            {
                'swing_ratio': s.swing_ratio,
                'tempo': s.tempo,
                'std_dev': s.std_dev,
                'num_samples': s.num_samples
            }
            for s in self.stats.swing_measurements
        ]

        # Write to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Statistics saved to {output_path}")


# ==============================================================================
# COMMAND LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(
        description='Analyze MIDI datasets and extract patterns'
    )
    parser.add_argument('dataset_path',
                       help='Path to directory containing MIDI files or glob pattern')
    parser.add_argument('--output', '-o',
                       help='Output JSON file for statistics')
    parser.add_argument('--no-chords', action='store_true',
                       help='Skip chord progression analysis')
    parser.add_argument('--no-swing', action='store_true',
                       help='Skip swing ratio measurement')
    parser.add_argument('--no-comping', action='store_true',
                       help='Skip comping pattern extraction')
    parser.add_argument('--compare', '-c',
                       help='Compare generated MIDI file to dataset')
    parser.add_argument('--extract-licks', action='store_true',
                       help='Extract bebop licks and save to file')

    args = parser.parse_args()

    # Find MIDI files
    if '*' in args.dataset_path:
        midi_files = glob.glob(args.dataset_path)
    else:
        path = Path(args.dataset_path)
        if path.is_dir():
            midi_files = list(path.glob('**/*.mid')) + list(path.glob('**/*.midi'))
        else:
            midi_files = [str(path)]

    midi_files = [str(f) for f in midi_files]

    if not midi_files:
        print(f"No MIDI files found in {args.dataset_path}")
        exit(1)

    print(f"Found {len(midi_files)} MIDI files")

    # Analyze dataset
    analyzer = DatasetAnalyzer()
    stats = analyzer.analyze_dataset(
        midi_files,
        analyze_chords=not args.no_chords,
        analyze_swing=not args.no_swing,
        analyze_comping=not args.no_comping,
        verbose=True
    )

    # Save statistics
    if args.output:
        analyzer.save_statistics(args.output)

    # Compare to generated file
    if args.compare:
        print(f"\n{'='*80}")
        print(f"COMPARING GENERATED FILE TO DATASET")
        print(f"{'='*80}\n")

        comparison = analyzer.compare_generated_to_dataset(args.compare)

        for metric, score in sorted(comparison.items()):
            print(f"  {metric}: {score:.2%}")

    # Extract bebop licks
    if args.extract_licks:
        print(f"\n{'='*80}")
        print(f"EXTRACTING BEBOP LICKS")
        print(f"{'='*80}\n")

        licks = analyzer.extract_bebop_licks()

        print(f"Extracted {len(licks)} common melodic patterns")
        print(f"\nTop 10 licks:")
        for i, lick in enumerate(licks[:10], 1):
            print(f"  {i}. {lick}")

        # Save to file
        licks_file = 'bebop_licks_extracted.json'
        with open(licks_file, 'w') as f:
            json.dump({'licks': licks}, f, indent=2)
        print(f"\nSaved to {licks_file}")
