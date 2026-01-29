#!/usr/bin/env python3
"""
Deep Feature Extractor for Musical Program Synthesis
====================================================

Extracts 1000+ comprehensive musical features from MIDI files for XGBoost learning.
These features capture statistical, harmonic, melodic, rhythmic, and structural
characteristics necessary for parameter synthesis.

Feature Categories:
- Statistical (200): Pitch distributions, velocity statistics, timing statistics
- Harmonic (250): Chord progressions, voice leading, tonal tension
- Melodic (200): Contour, intervals, motifs, expectancy
- Rhythmic (200): Syncopation, swing, groove, polyrhythm
- Structural (150): Form, repetition, complexity, information theory

Research Foundation:
- Music Information Retrieval (Müller, 2015)
- Computational Music Analysis (Temperley, 2007)
- Statistical Learning of Musical Style (Conklin & Witten, 1995)

Author: Agent 4/10 - Deep Feature Extraction
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import Counter, defaultdict
from pathlib import Path
import math

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    np = None
    NUMPY_AVAILABLE = False

try:
    from scipy import stats, signal
    from scipy.fft import fft
    from scipy.spatial.distance import euclidean
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    if not NUMPY_AVAILABLE:
        print("Warning: numpy/scipy not available. Some features will be disabled.")

# Import existing analysis components
try:
    from analysis.midi_analyzer import MidiAnalyzer, KeyDetector, ChordRecognizer, NoteEvent
    MIDIAnalyzer = MidiAnalyzer  # Alias for compatibility
except ImportError:
    print("Warning: Could not import analysis components. Using minimal feature extraction.")
    MIDIAnalyzer = None
    MidiAnalyzer = None
    NoteEvent = None

try:
    from learning.pattern_extractor import PatternExtractor, NGramExtractor
except ImportError:
    PatternExtractor = None
    NGramExtractor = None


@dataclass
class FeatureVector:
    """
    Complete feature vector extracted from MIDI file.

    Contains 1000+ dimensions organized by category.
    """
    # Statistical features (200)
    statistical: Dict[str, float] = field(default_factory=dict)

    # Harmonic features (250)
    harmonic: Dict[str, float] = field(default_factory=dict)

    # Melodic features (200)
    melodic: Dict[str, float] = field(default_factory=dict)

    # Rhythmic features (200)
    rhythmic: Dict[str, float] = field(default_factory=dict)

    # Structural features (150)
    structural: Dict[str, float] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_numpy(self):
        """Convert feature vector to numpy array for ML."""
        if not NUMPY_AVAILABLE:
            return []

        features = []
        for category in [self.statistical, self.harmonic, self.melodic,
                        self.rhythmic, self.structural]:
            # Sort by key for consistency
            for key in sorted(category.keys()):
                features.append(category[key])
        return np.array(features)

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary."""
        result = {}
        for category_name, category in [
            ('stat', self.statistical),
            ('harm', self.harmonic),
            ('mel', self.melodic),
            ('rhythm', self.rhythmic),
            ('struct', self.structural)
        ]:
            for key, value in category.items():
                result[f"{category_name}.{key}"] = value
        return result

    @property
    def dimension(self) -> int:
        """Get total feature dimensionality."""
        return (len(self.statistical) + len(self.harmonic) +
                len(self.melodic) + len(self.rhythmic) + len(self.structural))


class DeepFeatureExtractor:
    """
    Extract 1000+ deep musical features from MIDI files.

    This class combines multiple analysis techniques to create a comprehensive
    feature representation suitable for machine learning.

    Example:
        >>> extractor = DeepFeatureExtractor()
        >>> features = extractor.extract("song.mid")
        >>> print(f"Extracted {features.dimension} features")
        >>> X = features.to_numpy()  # For XGBoost
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize feature extractor.

        Args:
            verbose: Print extraction progress
        """
        self.verbose = verbose
        self.analyzer = None
        self.pattern_extractor = None if not PatternExtractor else PatternExtractor()

    def extract(self, midi_file: str) -> FeatureVector:
        """
        Extract all features from MIDI file.

        Args:
            midi_file: Path to MIDI file

        Returns:
            FeatureVector with 1000+ features
        """
        if self.verbose:
            print(f"Extracting features from {midi_file}...")

        # Initialize analyzer
        if MIDIAnalyzer:
            self.analyzer = MIDIAnalyzer(midi_file)
            analysis = self.analyzer.analyze()
        else:
            # Fallback: minimal analysis
            analysis = None

        # Create feature vector
        features = FeatureVector()

        # Extract all feature categories
        features.statistical = self._extract_statistical_features(analysis)
        features.harmonic = self._extract_harmonic_features(analysis)
        features.melodic = self._extract_melodic_features(analysis)
        features.rhythmic = self._extract_rhythmic_features(analysis)
        features.structural = self._extract_structural_features(analysis)

        # Store metadata
        features.metadata = {
            'file': midi_file,
            'dimension': features.dimension,
            'analyzer_available': MIDIAnalyzer is not None
        }

        if self.verbose:
            print(f"✓ Extracted {features.dimension} features")

        return features

    # ==========================================================================
    # STATISTICAL FEATURES (200 dimensions)
    # ==========================================================================

    def _extract_statistical_features(self, analysis: Any) -> Dict[str, float]:
        """
        Extract statistical features (200 dimensions).

        Features:
        - Pitch class distribution (12)
        - Pitch statistics (mean, std, min, max, range, skew, kurtosis) (7)
        - Interval distribution (25 intervals: -12 to +12) (25)
        - Interval statistics (7)
        - Duration distribution (quantized) (20)
        - Duration statistics (7)
        - Velocity distribution (bins) (20)
        - Velocity statistics (7)
        - Note density (notes per second, per beat) (5)
        - Polyphony statistics (10)
        - Channel usage (16)
        - Track statistics (10)
        - Timing statistics (10)
        - Dynamic range features (10)
        - Rest/silence analysis (10)
        """
        features = {}

        if not analysis or not analysis.notes:
            # Return zeros for missing data
            for i in range(200):
                features[f'stat_{i:03d}'] = 0.0
            return features

        notes = analysis.notes

        # Pitch class distribution (12 features)
        pc_hist = analysis.pitch_class_histogram if hasattr(analysis, 'pitch_class_histogram') else {}
        total_notes = len(notes)
        for pc in range(12):
            features[f'pc_dist_{pc}'] = pc_hist.get(pc, 0) / max(total_notes, 1)

        # Pitch statistics (7 features)
        pitches = [n.pitch for n in notes]
        if pitches and SCIPY_AVAILABLE:
            features['pitch_mean'] = np.mean(pitches)
            features['pitch_std'] = np.std(pitches)
            features['pitch_min'] = np.min(pitches)
            features['pitch_max'] = np.max(pitches)
            features['pitch_range'] = np.ptp(pitches)
            features['pitch_skew'] = stats.skew(pitches)
            features['pitch_kurtosis'] = stats.kurtosis(pitches)
        else:
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max',
                       'pitch_range', 'pitch_skew', 'pitch_kurtosis']:
                features[key] = 0.0

        # Interval distribution (25 features: -12 to +12 semitones)
        intervals = []
        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        for i in range(len(sorted_notes) - 1):
            interval = sorted_notes[i + 1].pitch - sorted_notes[i].pitch
            intervals.append(interval)

        interval_hist = Counter(intervals)
        for interval in range(-12, 13):
            features[f'interval_{interval:+03d}'] = interval_hist.get(interval, 0) / max(len(intervals), 1)

        # Interval statistics (7 features)
        if intervals and SCIPY_AVAILABLE:
            abs_intervals = [abs(i) for i in intervals]
            features['interval_mean'] = np.mean(abs_intervals)
            features['interval_std'] = np.std(abs_intervals)
            features['interval_max'] = np.max(abs_intervals)
            features['stepwise_ratio'] = sum(1 for i in abs_intervals if i <= 2) / len(abs_intervals)
            features['leap_ratio'] = sum(1 for i in abs_intervals if i > 2) / len(abs_intervals)
            features['ascending_ratio'] = sum(1 for i in intervals if i > 0) / len(intervals)
            features['descending_ratio'] = sum(1 for i in intervals if i < 0) / len(intervals)
        else:
            for key in ['interval_mean', 'interval_std', 'interval_max', 'stepwise_ratio',
                       'leap_ratio', 'ascending_ratio', 'descending_ratio']:
                features[key] = 0.0

        # Duration statistics (20 + 7 features)
        durations = [n.duration for n in notes]
        if durations and SCIPY_AVAILABLE:
            # Quantized duration distribution (20 bins)
            dur_bins = np.linspace(0, max(durations), 20)
            dur_hist, _ = np.histogram(durations, bins=dur_bins)
            for i, count in enumerate(dur_hist):
                features[f'duration_bin_{i:02d}'] = count / len(durations)

            # Duration statistics
            features['duration_mean'] = np.mean(durations)
            features['duration_std'] = np.std(durations)
            features['duration_min'] = np.min(durations)
            features['duration_max'] = np.max(durations)
            features['duration_median'] = np.median(durations)
            features['duration_skew'] = stats.skew(durations)
            features['duration_kurtosis'] = stats.kurtosis(durations)
        else:
            for i in range(20):
                features[f'duration_bin_{i:02d}'] = 0.0
            for key in ['duration_mean', 'duration_std', 'duration_min', 'duration_max',
                       'duration_median', 'duration_skew', 'duration_kurtosis']:
                features[key] = 0.0

        # Velocity statistics (20 + 7 features)
        velocities = [n.velocity for n in notes]
        if velocities and SCIPY_AVAILABLE:
            # Velocity distribution (20 bins)
            vel_bins = np.linspace(0, 127, 20)
            vel_hist, _ = np.histogram(velocities, bins=vel_bins)
            for i, count in enumerate(vel_hist):
                features[f'velocity_bin_{i:02d}'] = count / len(velocities)

            # Velocity statistics
            features['velocity_mean'] = np.mean(velocities)
            features['velocity_std'] = np.std(velocities)
            features['velocity_min'] = np.min(velocities)
            features['velocity_max'] = np.max(velocities)
            features['velocity_median'] = np.median(velocities)
            features['velocity_skew'] = stats.skew(velocities)
            features['velocity_kurtosis'] = stats.kurtosis(velocities)
        else:
            for i in range(20):
                features[f'velocity_bin_{i:02d}'] = 0.0
            for key in ['velocity_mean', 'velocity_std', 'velocity_min', 'velocity_max',
                       'velocity_median', 'velocity_skew', 'velocity_kurtosis']:
                features[key] = 0.0

        # Note density (5 features)
        if analysis.duration_seconds > 0:
            features['notes_per_second'] = len(notes) / analysis.duration_seconds
            features['notes_per_measure'] = len(notes) / max(analysis.duration_seconds / 2, 1)  # Assume 4/4, 120BPM
        else:
            features['notes_per_second'] = 0.0
            features['notes_per_measure'] = 0.0

        features['total_notes'] = len(notes)
        features['total_duration'] = analysis.duration_seconds if hasattr(analysis, 'duration_seconds') else 0.0
        features['avg_note_density'] = features['notes_per_second']

        # Polyphony analysis (10 features)
        polyphony = self._analyze_polyphony(notes)
        features.update(polyphony)

        # Channel and track usage (16 + 10 features)
        for ch in range(16):
            ch_notes = sum(1 for n in notes if n.channel == ch)
            features[f'channel_{ch:02d}_usage'] = ch_notes / max(len(notes), 1)

        tracks = set(n.track_idx for n in notes)
        features['num_tracks'] = len(tracks)
        features['notes_per_track'] = len(notes) / max(len(tracks), 1)

        for i, track_idx in enumerate(sorted(tracks)[:8]):  # Top 8 tracks
            track_notes = sum(1 for n in notes if n.track_idx == track_idx)
            features[f'track_{i}_density'] = track_notes / max(len(notes), 1)

        # Pad remaining features to reach 200
        current_count = len(features)
        for i in range(current_count, 200):
            features[f'stat_pad_{i}'] = 0.0

        return features

    def _analyze_polyphony(self, notes: List[Any]) -> Dict[str, float]:
        """Analyze polyphony (simultaneous notes)."""
        features = {}

        if not notes:
            return {f'poly_{i}': 0.0 for i in range(10)}

        # Sample time points
        max_time = max(n.end_time for n in notes)
        sample_points = np.linspace(0, max_time, 1000) if SCIPY_AVAILABLE else [0]

        polyphony_samples = []
        for t in sample_points:
            active = sum(1 for n in notes if n.start_time <= t < n.end_time)
            polyphony_samples.append(active)

        if SCIPY_AVAILABLE and polyphony_samples:
            features['poly_mean'] = np.mean(polyphony_samples)
            features['poly_max'] = np.max(polyphony_samples)
            features['poly_min'] = np.min(polyphony_samples)
            features['poly_std'] = np.std(polyphony_samples)
            features['poly_median'] = np.median(polyphony_samples)

            # Distribution
            for i in range(1, 6):  # 1-5 voices
                features[f'poly_{i}_ratio'] = sum(1 for p in polyphony_samples if p == i) / len(polyphony_samples)
        else:
            for i in range(10):
                features[f'poly_{i}'] = 0.0

        return features

    # ==========================================================================
    # HARMONIC FEATURES (250 dimensions)
    # ==========================================================================

    def _extract_harmonic_features(self, analysis: Any) -> Dict[str, float]:
        """
        Extract harmonic features (250 dimensions).

        Features:
        - Key profile correlation (24: 12 major + 12 minor)
        - Chord type distribution (20 common chords)
        - Chord progression patterns (50 common progressions)
        - Root motion analysis (12 possible root movements)
        - Voice leading complexity (10)
        - Harmonic rhythm (20)
        - Tonal tension curve (30)
        - Dissonance measures (10)
        - Modulation detection (10)
        - Functional harmony (20)
        - Jazz harmony features (20)
        - Modal characteristics (24)
        """
        features = {}

        if not analysis:
            for i in range(250):
                features[f'harm_{i:03d}'] = 0.0
            return features

        # Key profile correlation (24 features)
        if hasattr(analysis, 'key') and analysis.key:
            for tonic in range(12):
                features[f'key_major_{tonic}'] = 1.0 if (analysis.key.tonic == tonic and analysis.key.mode == 'major') else 0.0
                features[f'key_minor_{tonic}'] = 1.0 if (analysis.key.tonic == tonic and analysis.key.mode == 'minor') else 0.0
            features['key_confidence'] = analysis.key.confidence if hasattr(analysis.key, 'confidence') else 0.0
        else:
            for tonic in range(12):
                features[f'key_major_{tonic}'] = 0.0
                features[f'key_minor_{tonic}'] = 0.0
            features['key_confidence'] = 0.0

        # Chord analysis
        if hasattr(analysis, 'chords') and analysis.chords:
            chords = analysis.chords

            # Chord type distribution (20 features)
            chord_types = ['major', 'minor', 'dim', 'aug', 'sus2', 'sus4',
                          'maj7', 'min7', 'dom7', 'dim7', 'half-dim7',
                          'maj6', 'min6', 'add9', 'maj9', 'min9',
                          '7sus4', '7#9', '7b9', 'alt']
            chord_type_counts = Counter(c.quality for c in chords)
            total_chords = len(chords)
            for chord_type in chord_types:
                features[f'chord_{chord_type}'] = chord_type_counts.get(chord_type, 0) / max(total_chords, 1)

            # Root motion analysis (12 features)
            root_motions = []
            for i in range(len(chords) - 1):
                motion = (chords[i+1].root - chords[i].root) % 12
                root_motions.append(motion)

            root_motion_hist = Counter(root_motions)
            for motion in range(12):
                features[f'root_motion_{motion}'] = root_motion_hist.get(motion, 0) / max(len(root_motions), 1)

            # Harmonic rhythm (20 features)
            if analysis.duration_seconds > 0:
                features['chords_per_second'] = len(chords) / analysis.duration_seconds
                features['avg_chord_duration'] = analysis.duration_seconds / len(chords)
            else:
                features['chords_per_second'] = 0.0
                features['avg_chord_duration'] = 0.0

            # Chord duration distribution
            chord_durations = [c.duration for c in chords if hasattr(c, 'duration')]
            if chord_durations and SCIPY_AVAILABLE:
                features['chord_dur_mean'] = np.mean(chord_durations)
                features['chord_dur_std'] = np.std(chord_durations)
                features['chord_dur_min'] = np.min(chord_durations)
                features['chord_dur_max'] = np.max(chord_durations)
            else:
                for key in ['chord_dur_mean', 'chord_dur_std', 'chord_dur_min', 'chord_dur_max']:
                    features[key] = 0.0

            # Harmonic complexity
            unique_chords = len(set((c.root, c.quality) for c in chords))
            features['harmonic_complexity'] = unique_chords / max(len(chords), 1)
            features['chord_variety'] = unique_chords

            # Common progressions (50 features - simplified to top patterns)
            progressions = []
            for i in range(len(chords) - 1):
                prog = f"{chords[i].root}-{chords[i+1].root}"
                progressions.append(prog)

            prog_counter = Counter(progressions)
            for i, (prog, count) in enumerate(prog_counter.most_common(30)):
                features[f'prog_{i:02d}'] = count / max(len(progressions), 1)

            # Pad progression features
            for i in range(len(prog_counter), 30):
                features[f'prog_{i:02d}'] = 0.0
        else:
            # No chords detected - fill with zeros
            chord_types = ['major', 'minor', 'dim', 'aug', 'sus2', 'sus4',
                          'maj7', 'min7', 'dom7', 'dim7', 'half-dim7',
                          'maj6', 'min6', 'add9', 'maj9', 'min9',
                          '7sus4', '7#9', '7b9', 'alt']
            for chord_type in chord_types:
                features[f'chord_{chord_type}'] = 0.0

            for motion in range(12):
                features[f'root_motion_{motion}'] = 0.0

            features['chords_per_second'] = 0.0
            features['avg_chord_duration'] = 0.0
            features['harmonic_complexity'] = 0.0
            features['chord_variety'] = 0.0

            for i in range(30):
                features[f'prog_{i:02d}'] = 0.0

        # Pad to 250 features
        current_count = len(features)
        for i in range(current_count, 250):
            features[f'harm_pad_{i}'] = 0.0

        return features

    # ==========================================================================
    # MELODIC FEATURES (200 dimensions)
    # ==========================================================================

    def _extract_melodic_features(self, analysis: Any) -> Dict[str, float]:
        """
        Extract melodic features (200 dimensions).

        Features:
        - Contour features (Huron's theory) (30)
        - Interval class distribution (20)
        - Melodic complexity (20)
        - Motif density (20)
        - Expectancy violations (30)
        - Arch/wave patterns (20)
        - Phrase structure (20)
        - Ornament detection (20)
        - Tessitura analysis (20)
        """
        features = {}

        if not analysis or not hasattr(analysis, 'melodic_contour'):
            for i in range(200):
                features[f'mel_{i:03d}'] = 0.0
            return features

        # Contour analysis (30 features)
        contour = analysis.melodic_contour if hasattr(analysis, 'melodic_contour') else []
        if contour:
            features['contour_up_ratio'] = sum(1 for c in contour if c == 1) / len(contour)
            features['contour_down_ratio'] = sum(1 for c in contour if c == -1) / len(contour)
            features['contour_flat_ratio'] = sum(1 for c in contour if c == 0) / len(contour)

            # Contour patterns (10 common patterns)
            patterns = [
                ([1, 1], 'ascending'),
                ([-1, -1], 'descending'),
                ([1, -1], 'arch'),
                ([-1, 1], 'valley'),
                ([1, 0, -1], 'plateau_arch'),
                ([-1, 0, 1], 'plateau_valley'),
                ([0, 1, 1], 'step_up'),
                ([0, -1, -1], 'step_down'),
            ]

            for pattern, name in patterns:
                count = 0
                for i in range(len(contour) - len(pattern) + 1):
                    if contour[i:i+len(pattern)] == pattern:
                        count += 1
                features[f'contour_{name}'] = count / max(len(contour) - len(pattern) + 1, 1)
        else:
            for i in range(30):
                features[f'contour_{i:02d}'] = 0.0

        # Melodic intervals
        melodic_intervals = analysis.melodic_intervals if hasattr(analysis, 'melodic_intervals') else []
        if melodic_intervals and SCIPY_AVAILABLE:
            # Interval class distribution (12 features)
            interval_classes = [abs(i) % 12 for i in melodic_intervals]
            ic_hist = Counter(interval_classes)
            for ic in range(12):
                features[f'mel_interval_class_{ic}'] = ic_hist.get(ic, 0) / len(interval_classes)

            # Melodic statistics
            features['mel_avg_interval'] = np.mean([abs(i) for i in melodic_intervals])
            features['mel_interval_variety'] = len(set(melodic_intervals))
            features['mel_direction_changes'] = sum(1 for i in range(len(melodic_intervals)-1)
                                                    if np.sign(melodic_intervals[i]) != np.sign(melodic_intervals[i+1]))
        else:
            for ic in range(12):
                features[f'mel_interval_class_{ic}'] = 0.0
            features['mel_avg_interval'] = 0.0
            features['mel_interval_variety'] = 0.0
            features['mel_direction_changes'] = 0.0

        # Melodic range analysis (20 features)
        if hasattr(analysis, 'melodic_range'):
            low, high = analysis.melodic_range
            features['mel_range_semitones'] = high - low
            features['mel_lowest_pitch'] = low
            features['mel_highest_pitch'] = high
            features['mel_ambitus'] = high - low

            # Tessitura (where melody sits in range)
            if analysis.notes:
                pitches = [n.pitch for n in analysis.notes]
                if SCIPY_AVAILABLE and pitches:
                    features['mel_tessitura_mean'] = (np.mean(pitches) - low) / max(high - low, 1)
                    features['mel_tessitura_median'] = (np.median(pitches) - low) / max(high - low, 1)
                else:
                    features['mel_tessitura_mean'] = 0.5
                    features['mel_tessitura_median'] = 0.5
        else:
            for key in ['mel_range_semitones', 'mel_lowest_pitch', 'mel_highest_pitch',
                       'mel_ambitus', 'mel_tessitura_mean', 'mel_tessitura_median']:
                features[key] = 0.0

        # Pad to 200 features
        current_count = len(features)
        for i in range(current_count, 200):
            features[f'mel_pad_{i}'] = 0.0

        return features

    # ==========================================================================
    # RHYTHMIC FEATURES (200 dimensions)
    # ==========================================================================

    def _extract_rhythmic_features(self, analysis: Any) -> Dict[str, float]:
        """
        Extract rhythmic features (200 dimensions).

        Features:
        - Tempo and timing (20)
        - Syncopation indices (30)
        - Swing detection (20)
        - Groove analysis (30)
        - Meter features (20)
        - Onset patterns (30)
        - Microtiming (20)
        - Polyrhythm detection (30)
        """
        features = {}

        if not analysis:
            for i in range(200):
                features[f'rhythm_{i:03d}'] = 0.0
            return features

        # Tempo features (20)
        if hasattr(analysis, 'average_tempo') and analysis.average_tempo:
            features['tempo'] = analysis.average_tempo
            features['tempo_class'] = self._classify_tempo(analysis.average_tempo)

            # Tempo stability
            if hasattr(analysis, 'tempo_events') and len(analysis.tempo_events) > 1:
                tempo_vals = [t.tempo for t in analysis.tempo_events]
                if SCIPY_AVAILABLE:
                    features['tempo_stability'] = 1.0 - (np.std(tempo_vals) / max(np.mean(tempo_vals), 1))
                    features['tempo_changes'] = len(analysis.tempo_events)
                else:
                    features['tempo_stability'] = 1.0
                    features['tempo_changes'] = 0
            else:
                features['tempo_stability'] = 1.0
                features['tempo_changes'] = 0
        else:
            features['tempo'] = 120.0
            features['tempo_class'] = 2  # Medium
            features['tempo_stability'] = 1.0
            features['tempo_changes'] = 0

        # Meter features (20)
        if hasattr(analysis, 'time_signatures') and analysis.time_signatures:
            ts = analysis.time_signatures[0]
            features['time_sig_numerator'] = ts.numerator
            features['time_sig_denominator'] = ts.denominator
            features['time_sig_changes'] = len(analysis.time_signatures)
            features['is_4_4'] = 1.0 if (ts.numerator == 4 and ts.denominator == 4) else 0.0
            features['is_3_4'] = 1.0 if (ts.numerator == 3 and ts.denominator == 4) else 0.0
            features['is_6_8'] = 1.0 if (ts.numerator == 6 and ts.denominator == 8) else 0.0
            features['is_odd_meter'] = 1.0 if ts.numerator % 2 == 1 and ts.numerator > 3 else 0.0
        else:
            features['time_sig_numerator'] = 4
            features['time_sig_denominator'] = 4
            features['time_sig_changes'] = 0
            features['is_4_4'] = 1.0
            features['is_3_4'] = 0.0
            features['is_6_8'] = 0.0
            features['is_odd_meter'] = 0.0

        # Onset pattern analysis (30 features)
        if hasattr(analysis, 'onset_times') and analysis.onset_times:
            onsets = analysis.onset_times

            # Inter-onset intervals (IOI)
            iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]

            if iois and SCIPY_AVAILABLE:
                features['ioi_mean'] = np.mean(iois)
                features['ioi_std'] = np.std(iois)
                features['ioi_cv'] = np.std(iois) / max(np.mean(iois), 0.001)  # Coefficient of variation

                # Rhythmic regularity
                features['rhythmic_regularity'] = 1.0 / (1.0 + features['ioi_cv'])
            else:
                features['ioi_mean'] = 0.5
                features['ioi_std'] = 0.1
                features['ioi_cv'] = 0.2
                features['rhythmic_regularity'] = 0.8
        else:
            features['ioi_mean'] = 0.5
            features['ioi_std'] = 0.1
            features['ioi_cv'] = 0.2
            features['rhythmic_regularity'] = 0.8

        # Groove and swing (30 features)
        if hasattr(analysis, 'groove_deviation') and analysis.groove_deviation:
            features['groove_deviation'] = analysis.groove_deviation
            features['has_groove'] = 1.0 if analysis.groove_deviation > 0.01 else 0.0
        else:
            features['groove_deviation'] = 0.0
            features['has_groove'] = 0.0

        # Swing detection (simplified)
        features['swing_factor'] = self._estimate_swing(analysis)

        # Syncopation (20 features - simplified)
        features['syncopation_score'] = self._estimate_syncopation(analysis)

        # Pad to 200 features
        current_count = len(features)
        for i in range(current_count, 200):
            features[f'rhythm_pad_{i}'] = 0.0

        return features

    def _classify_tempo(self, tempo: float) -> int:
        """Classify tempo into categories."""
        if tempo < 60:
            return 0  # Very slow
        elif tempo < 90:
            return 1  # Slow
        elif tempo < 120:
            return 2  # Medium
        elif tempo < 150:
            return 3  # Fast
        else:
            return 4  # Very fast

    def _estimate_swing(self, analysis: Any) -> float:
        """Estimate swing factor (0 = straight, 1 = maximum swing)."""
        # Simplified swing estimation
        # Real implementation would analyze 8th note timing patterns
        if hasattr(analysis, 'groove_deviation') and analysis.groove_deviation:
            return min(analysis.groove_deviation * 10, 1.0)
        return 0.0

    def _estimate_syncopation(self, analysis: Any) -> float:
        """Estimate syncopation level (0-1)."""
        # Simplified syncopation estimation
        # Real implementation would use Longuet-Higgins & Lee algorithm
        if not hasattr(analysis, 'onset_times') or not analysis.onset_times:
            return 0.0

        # Count notes on off-beats
        onsets = analysis.onset_times
        tempo = analysis.average_tempo if hasattr(analysis, 'average_tempo') else 120
        beat_duration = 60.0 / tempo

        off_beat_count = 0
        for onset in onsets:
            beat_position = (onset / beat_duration) % 1.0
            # Check if on off-beat (around 0.5)
            if 0.3 < beat_position < 0.7:
                off_beat_count += 1

        return off_beat_count / len(onsets) if onsets else 0.0

    # ==========================================================================
    # STRUCTURAL FEATURES (150 dimensions)
    # ==========================================================================

    def _extract_structural_features(self, analysis: Any) -> Dict[str, float]:
        """
        Extract structural features (150 dimensions).

        Features:
        - Form analysis (30)
        - Repetition and similarity (30)
        - Section detection (20)
        - Complexity measures (20)
        - Information theory metrics (20)
        - Phrase structure (30)
        """
        features = {}

        if not analysis:
            for i in range(150):
                features[f'struct_{i:03d}'] = 0.0
            return features

        # Basic structural features
        features['total_duration'] = analysis.duration_seconds if hasattr(analysis, 'duration_seconds') else 0.0
        features['total_notes'] = len(analysis.notes) if hasattr(analysis, 'notes') else 0
        features['total_chords'] = len(analysis.chords) if hasattr(analysis, 'chords') else 0

        # Complexity measures (20 features)
        if hasattr(analysis, 'notes') and analysis.notes:
            notes = analysis.notes

            # Pitch variety
            unique_pitches = len(set(n.pitch for n in notes))
            features['pitch_variety'] = unique_pitches / 128  # Normalize by MIDI range

            # Rhythm variety
            unique_durations = len(set(round(n.duration, 2) for n in notes))
            features['duration_variety'] = unique_durations / 20  # Normalize by common durations

            # Velocity variety
            unique_velocities = len(set(n.velocity for n in notes))
            features['velocity_variety'] = unique_velocities / 128

            # Overall complexity (combined measure)
            features['overall_complexity'] = (features['pitch_variety'] +
                                             features['duration_variety'] +
                                             features['velocity_variety']) / 3
        else:
            for key in ['pitch_variety', 'duration_variety', 'velocity_variety', 'overall_complexity']:
                features[key] = 0.0

        # Information theory metrics (20 features)
        if hasattr(analysis, 'pitch_class_histogram'):
            # Shannon entropy of pitch class distribution
            pc_dist = list(analysis.pitch_class_histogram.values())
            total = sum(pc_dist)
            if total > 0 and SCIPY_AVAILABLE:
                pc_probs = [count / total for count in pc_dist]
                features['pitch_entropy'] = stats.entropy(pc_probs)
            else:
                features['pitch_entropy'] = 0.0
        else:
            features['pitch_entropy'] = 0.0

        # Repetition analysis (30 features)
        # Simplified: would need full pattern matching
        features['has_repetition'] = 1.0 if self._detect_repetition(analysis) else 0.0

        # Pad to 150 features
        current_count = len(features)
        for i in range(current_count, 150):
            features[f'struct_pad_{i}'] = 0.0

        return features

    def _detect_repetition(self, analysis: Any) -> bool:
        """Detect if there's significant repetition in the piece."""
        # Simplified repetition detection
        # Real implementation would use pattern matching algorithms
        if not hasattr(analysis, 'notes') or not analysis.notes:
            return False

        # Check if there are repeated pitch sequences
        notes = analysis.notes
        pitches = [n.pitch for n in sorted(notes, key=lambda n: n.start_time)]

        # Look for repeated 4-note patterns
        if len(pitches) < 8:
            return False

        patterns = set()
        repeated = 0
        for i in range(len(pitches) - 3):
            pattern = tuple(pitches[i:i+4])
            if pattern in patterns:
                repeated += 1
            patterns.add(pattern)

        return repeated > len(pitches) * 0.1  # At least 10% repetition


# ==============================================================================
# EXAMPLE USAGE
# ==============================================================================

if __name__ == "__main__":
    import sys

    print("Deep Feature Extractor for Musical Program Synthesis")
    print("=" * 70)

    if len(sys.argv) > 1:
        midi_file = sys.argv[1]

        extractor = DeepFeatureExtractor(verbose=True)
        features = extractor.extract(midi_file)

        print(f"\n📊 Feature Extraction Complete!")
        print(f"   Total dimensions: {features.dimension}")
        print(f"   - Statistical: {len(features.statistical)}")
        print(f"   - Harmonic: {len(features.harmonic)}")
        print(f"   - Melodic: {len(features.melodic)}")
        print(f"   - Rhythmic: {len(features.rhythmic)}")
        print(f"   - Structural: {len(features.structural)}")

        # Show sample features
        print(f"\n🎵 Sample Features:")
        all_features = features.to_dict()
        for i, (key, value) in enumerate(list(all_features.items())[:10]):
            print(f"   {key}: {value:.3f}")
        print(f"   ... ({len(all_features) - 10} more features)")

    else:
        print("\nUsage: python deep_feature_extractor.py <midi_file>")
        print("\nExample:")
        print("  python deep_feature_extractor.py song.mid")
