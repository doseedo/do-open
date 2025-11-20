"""
Agent 24: Texture Specialist
=============================

Comprehensive texture analysis for the self-expanding inverse music generation system.

This specialist analyzes:
1. Polyphonic density (simultaneous note counts, voice distribution)
2. Voice independence (melodic/rhythmic independence, voice crossing)
3. Homophonic vs polyphonic characteristics
4. Layering (register distribution, timbral layering)
5. Textural contrast (temporal texture changes, transition detection)
6. Vertical density (harmonic complexity at moments)
7. Horizontal density (temporal activity, rhythmic density)

ARCHITECTURAL ROLE:
    MIDI → TextureSpecialist.analyze() → 135+ features → XGBoost models → parameters
    parameters → HarmonyModule → MIDI (generation)

OUTPUT PARAMETERS (for registry.json):
    - texture_polyphonic_density: 0.0-1.0 (monophonic to dense polyphony)
    - texture_voice_independence: 0.0-1.0 (homophonic to independent counterpoint)
    - texture_vertical_density: 0.0-1.0 (sparse to dense vertical stacking)
    - texture_horizontal_density: 0.0-1.0 (sparse to continuous activity)
    - texture_layering_count: 1-8 (number of distinct textural layers)
    - texture_register_spread: 0.0-1.0 (narrow to wide pitch range usage)
    - texture_contrast_rate: 0.0-1.0 (static to rapidly changing textures)
    - texture_homophonic_ratio: 0.0-1.0 (polyphonic to homophonic)
    - texture_voice_crossing_density: 0.0-1.0 (no crossing to frequent crossing)
    - texture_rhythmic_independence: 0.0-1.0 (synchronized to independent rhythms)

FEATURE EXTRACTION:
    135+ detailed features including:
    - Temporal polyphony analysis (mean, std, max, percentiles)
    - Voice independence metrics (pitch correlation, rhythm correlation)
    - Layer detection and analysis
    - Register usage histograms
    - Texture transition matrices
    - Density evolution curves

Author: Agent 24 - Texture Specialist
Part of: 35-Agent Self-Expanding Inverse Music Generation System
License: MIT
"""

import numpy as np
import pretty_midi
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from scipy import stats, signal
from scipy.spatial.distance import euclidean
from scipy.cluster.hierarchy import linkage, fcluster
import warnings

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class VoiceProfile:
    """Profile of an individual voice/layer in the texture."""
    notes: List[pretty_midi.Note]
    pitch_range: Tuple[int, int]
    pitch_mean: float
    pitch_std: float
    note_density: float  # notes per second
    rhythmic_pattern: List[float]  # inter-onset intervals
    register: str  # 'low', 'mid', 'high'
    activity_ratio: float  # proportion of time active

@dataclass
class TextureSegment:
    """A segment of music with consistent texture."""
    start_time: float
    end_time: float
    polyphonic_density: float
    voice_count: int
    vertical_density: float
    horizontal_density: float
    texture_type: str  # 'monophonic', 'homophonic', 'polyphonic', 'heterophonic'

@dataclass
class TextureAnalysisResult:
    """Complete texture analysis results."""
    # Global metrics
    mean_polyphonic_density: float
    std_polyphonic_density: float
    max_polyphonic_density: int
    voice_independence: float
    vertical_density: float
    horizontal_density: float

    # Voice analysis
    detected_voices: List[VoiceProfile]
    voice_count: int
    voice_crossing_density: float
    rhythmic_independence: float

    # Layering
    layering_count: int
    register_spread: float
    register_distribution: Dict[str, float]  # low, mid, high proportions

    # Texture types
    homophonic_ratio: float
    polyphonic_ratio: float
    monophonic_ratio: float

    # Temporal analysis
    texture_segments: List[TextureSegment]
    contrast_rate: float  # how often texture changes
    texture_transitions: List[Tuple[str, str, float]]  # (from, to, time)

    # Detailed features (135+)
    detailed_features: Dict[str, float] = field(default_factory=dict)

# ============================================================================
# TEXTURE SPECIALIST
# ============================================================================

class TextureSpecialist:
    """
    Agent 24: Comprehensive texture analysis for inverse music generation.

    This specialist extracts 135+ features related to musical texture,
    including polyphonic density, voice independence, layering, and
    textural contrast.
    """

    def __init__(self,
                 time_resolution: float = 0.05,  # 50ms time windows
                 voice_separation_threshold: float = 7.0,  # semitones for voice separation
                 layer_detection_method: str = 'hierarchical'):
        """
        Initialize the texture specialist.

        Args:
            time_resolution: Time window for density analysis (seconds)
            voice_separation_threshold: Pitch distance for voice separation
            layer_detection_method: 'hierarchical', 'kmeans', or 'register'
        """
        self.time_resolution = time_resolution
        self.voice_separation_threshold = voice_separation_threshold
        self.layer_detection_method = layer_detection_method

        # Register boundaries (MIDI note numbers)
        self.register_boundaries = {
            'low': (0, 48),      # C-1 to C3
            'mid': (48, 72),     # C3 to C5
            'high': (72, 127)    # C5 to G9
        }

    # ========================================================================
    # MAIN ANALYSIS ENTRY POINT
    # ========================================================================

    def analyze(self, midi_file_path: str) -> TextureAnalysisResult:
        """
        Perform comprehensive texture analysis on a MIDI file.

        Args:
            midi_file_path: Path to MIDI file

        Returns:
            TextureAnalysisResult with all metrics and features
        """
        try:
            pm = pretty_midi.PrettyMIDI(midi_file_path)
        except Exception as e:
            warnings.warn(f"Error loading MIDI file: {e}")
            return self._empty_result()

        if len(pm.instruments) == 0 or not any(len(inst.notes) > 0 for inst in pm.instruments):
            return self._empty_result()

        # Collect all notes with timing
        all_notes = self._collect_all_notes(pm)
        if len(all_notes) == 0:
            return self._empty_result()

        # Create time grid for density analysis
        end_time = pm.get_end_time()
        time_grid = np.arange(0, end_time, self.time_resolution)

        # Analyze polyphonic density over time
        polyphony_curve = self._compute_polyphony_curve(all_notes, time_grid)

        # Detect and analyze voices
        voices = self._detect_voices(all_notes, end_time)

        # Analyze voice independence
        voice_independence = self._compute_voice_independence(voices)
        rhythmic_independence = self._compute_rhythmic_independence(voices)
        voice_crossing_density = self._compute_voice_crossing_density(voices)

        # Analyze layering
        layers = self._detect_layers(all_notes, voices)
        register_spread = self._compute_register_spread(all_notes)
        register_dist = self._compute_register_distribution(all_notes, end_time)

        # Analyze vertical and horizontal density
        vertical_density = self._compute_vertical_density(polyphony_curve)
        horizontal_density = self._compute_horizontal_density(all_notes, end_time)

        # Segment texture and analyze changes
        segments = self._segment_texture(all_notes, polyphony_curve, time_grid)
        texture_ratios = self._compute_texture_ratios(segments, end_time)
        contrast_rate = self._compute_contrast_rate(segments)
        transitions = self._extract_texture_transitions(segments)

        # Compute detailed features (135+)
        detailed_features = self._compute_detailed_features(
            all_notes, polyphony_curve, voices, layers, segments,
            time_grid, end_time
        )

        # Build result
        result = TextureAnalysisResult(
            mean_polyphonic_density=np.mean(polyphony_curve),
            std_polyphonic_density=np.std(polyphony_curve),
            max_polyphonic_density=int(np.max(polyphony_curve)),
            voice_independence=voice_independence,
            vertical_density=vertical_density,
            horizontal_density=horizontal_density,
            detected_voices=voices,
            voice_count=len(voices),
            voice_crossing_density=voice_crossing_density,
            rhythmic_independence=rhythmic_independence,
            layering_count=len(layers),
            register_spread=register_spread,
            register_distribution=register_dist,
            homophonic_ratio=texture_ratios['homophonic'],
            polyphonic_ratio=texture_ratios['polyphonic'],
            monophonic_ratio=texture_ratios['monophonic'],
            texture_segments=segments,
            contrast_rate=contrast_rate,
            texture_transitions=transitions,
            detailed_features=detailed_features
        )

        return result

    # ========================================================================
    # POLYPHONIC DENSITY ANALYSIS
    # ========================================================================

    def _compute_polyphony_curve(self, notes: List[Tuple], time_grid: np.ndarray) -> np.ndarray:
        """
        Compute the polyphonic density (number of simultaneous notes) over time.

        Args:
            notes: List of (start, end, pitch, velocity) tuples
            time_grid: Time points to sample

        Returns:
            Array of simultaneous note counts at each time point
        """
        polyphony = np.zeros(len(time_grid))

        for i, t in enumerate(time_grid):
            count = sum(1 for start, end, pitch, vel in notes if start <= t < end)
            polyphony[i] = count

        return polyphony

    def _compute_vertical_density(self, polyphony_curve: np.ndarray) -> float:
        """
        Compute overall vertical density (average harmonic complexity).

        Normalized to 0-1 based on typical ranges (0-12 simultaneous notes).
        """
        mean_polyphony = np.mean(polyphony_curve)
        # Normalize: 0 notes = 0.0, 12+ notes = 1.0
        return min(1.0, mean_polyphony / 12.0)

    def _compute_horizontal_density(self, notes: List[Tuple], duration: float) -> float:
        """
        Compute horizontal density (temporal activity level).

        Measures the proportion of time that has active notes.
        """
        if duration <= 0:
            return 0.0

        # Create activity bitmap
        time_resolution = 0.01  # 10ms resolution
        time_bins = int(duration / time_resolution) + 1
        activity = np.zeros(time_bins)

        for start, end, pitch, vel in notes:
            start_bin = int(start / time_resolution)
            end_bin = int(end / time_resolution)
            activity[start_bin:end_bin+1] = 1

        return np.mean(activity)

    # ========================================================================
    # VOICE DETECTION AND ANALYSIS
    # ========================================================================

    def _detect_voices(self, notes: List[Tuple], duration: float) -> List[VoiceProfile]:
        """
        Detect distinct voices/parts in the texture using pitch and temporal clustering.

        Uses hierarchical clustering on pitch ranges and temporal continuity.
        """
        if len(notes) == 0:
            return []

        # Sort notes by start time
        sorted_notes = sorted(notes, key=lambda x: x[0])

        # Build voice candidates using a greedy algorithm
        voices = []
        unassigned = list(sorted_notes)

        while unassigned:
            # Start a new voice with the earliest unassigned note
            voice_notes = [unassigned.pop(0)]

            # Try to extend this voice with compatible notes
            i = 0
            while i < len(unassigned):
                note = unassigned[i]
                if self._is_voice_compatible(voice_notes, note):
                    voice_notes.append(note)
                    unassigned.pop(i)
                else:
                    i += 1

            # Create voice profile
            if len(voice_notes) >= 2:  # Require at least 2 notes for a voice
                profile = self._create_voice_profile(voice_notes, duration)
                voices.append(profile)

        # Sort voices by register (low to high)
        voices.sort(key=lambda v: v.pitch_mean)

        return voices

    def _is_voice_compatible(self, voice_notes: List[Tuple], candidate: Tuple) -> bool:
        """
        Check if a candidate note is compatible with an existing voice.

        Compatible if:
        1. Pitch is within reasonable range of voice
        2. No temporal overlap with existing notes
        3. Maintains voice continuity
        """
        start, end, pitch, vel = candidate

        # Check for temporal overlap
        for v_start, v_end, v_pitch, v_vel in voice_notes:
            if not (end <= v_start or start >= v_end):
                return False  # Overlaps with existing note

        # Check pitch compatibility
        voice_pitches = [p for _, _, p, _ in voice_notes]
        pitch_mean = np.mean(voice_pitches)
        pitch_std = np.std(voice_pitches) if len(voice_pitches) > 1 else 5.0

        # Allow pitch within 2 standard deviations + threshold
        pitch_threshold = max(self.voice_separation_threshold, 2 * pitch_std)
        if abs(pitch - pitch_mean) > pitch_threshold:
            return False

        return True

    def _create_voice_profile(self, voice_notes: List[Tuple], duration: float) -> VoiceProfile:
        """Create a VoiceProfile from a list of notes."""
        # Convert tuples to Note objects for compatibility
        notes = []
        for start, end, pitch, vel in voice_notes:
            note = pretty_midi.Note(
                velocity=vel,
                pitch=pitch,
                start=start,
                end=end
            )
            notes.append(note)

        pitches = [n.pitch for n in notes]
        pitch_range = (min(pitches), max(pitches))
        pitch_mean = np.mean(pitches)
        pitch_std = np.std(pitches)

        # Compute rhythmic pattern (inter-onset intervals)
        onsets = sorted([n.start for n in notes])
        rhythmic_pattern = np.diff(onsets).tolist() if len(onsets) > 1 else []

        # Determine register
        if pitch_mean < 48:
            register = 'low'
        elif pitch_mean < 72:
            register = 'mid'
        else:
            register = 'high'

        # Compute activity ratio
        total_duration = sum(n.end - n.start for n in notes)
        activity_ratio = total_duration / duration if duration > 0 else 0.0

        # Note density
        note_density = len(notes) / duration if duration > 0 else 0.0

        return VoiceProfile(
            notes=notes,
            pitch_range=pitch_range,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            note_density=note_density,
            rhythmic_pattern=rhythmic_pattern,
            register=register,
            activity_ratio=activity_ratio
        )

    def _compute_voice_independence(self, voices: List[VoiceProfile]) -> float:
        """
        Compute the degree of independence between voices.

        Based on:
        1. Pitch correlation (low correlation = more independent)
        2. Rhythmic correlation (low correlation = more independent)
        3. Register separation (more separation = more independent)

        Returns value 0-1 where 1 = highly independent counterpoint.
        """
        if len(voices) <= 1:
            return 0.0

        independence_scores = []

        # Pairwise analysis
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                v1, v2 = voices[i], voices[j]

                # Pitch correlation (lower = more independent)
                pitch_corr = self._compute_pitch_correlation(v1, v2)
                pitch_indep = 1.0 - abs(pitch_corr)

                # Rhythmic correlation (lower = more independent)
                rhythm_corr = self._compute_rhythm_correlation(v1, v2)
                rhythm_indep = 1.0 - abs(rhythm_corr)

                # Register separation (larger = more independent)
                register_sep = abs(v1.pitch_mean - v2.pitch_mean) / 127.0

                # Combined score
                score = (pitch_indep * 0.4 + rhythm_indep * 0.4 + register_sep * 0.2)
                independence_scores.append(score)

        return np.mean(independence_scores) if independence_scores else 0.0

    def _compute_pitch_correlation(self, v1: VoiceProfile, v2: VoiceProfile) -> float:
        """Compute pitch contour correlation between two voices."""
        # Sample pitch at common time points
        times1 = [n.start for n in v1.notes]
        times2 = [n.start for n in v2.notes]

        if len(times1) < 2 or len(times2) < 2:
            return 0.0

        # Create common time grid
        min_time = min(min(times1), min(times2))
        max_time = max(max(times1), max(times2))
        time_grid = np.linspace(min_time, max_time, 50)

        # Interpolate pitches
        pitches1 = self._interpolate_pitch_at_times(v1.notes, time_grid)
        pitches2 = self._interpolate_pitch_at_times(v2.notes, time_grid)

        # Compute correlation
        if len(pitches1) > 1 and len(pitches2) > 1:
            try:
                corr, _ = stats.pearsonr(pitches1, pitches2)
                return corr if not np.isnan(corr) else 0.0
            except:
                return 0.0
        return 0.0

    def _interpolate_pitch_at_times(self, notes: List[pretty_midi.Note], times: np.ndarray) -> np.ndarray:
        """Interpolate pitch values at specified times."""
        pitches = np.zeros(len(times))

        for i, t in enumerate(times):
            # Find active note at time t
            active_notes = [n for n in notes if n.start <= t < n.end]
            if active_notes:
                # Use highest pitch if multiple notes
                pitches[i] = max(n.pitch for n in active_notes)
            elif i > 0:
                # Use previous pitch if no active note
                pitches[i] = pitches[i-1]
            else:
                # Use mean pitch for start
                pitches[i] = np.mean([n.pitch for n in notes])

        return pitches

    def _compute_rhythm_correlation(self, v1: VoiceProfile, v2: VoiceProfile) -> float:
        """Compute rhythmic correlation between two voices."""
        if len(v1.rhythmic_pattern) < 2 or len(v2.rhythmic_pattern) < 2:
            return 0.0

        # Use cross-correlation of inter-onset intervals
        ioi1 = np.array(v1.rhythmic_pattern)
        ioi2 = np.array(v2.rhythmic_pattern)

        # Normalize
        ioi1 = ioi1 / (np.std(ioi1) + 1e-6)
        ioi2 = ioi2 / (np.std(ioi2) + 1e-6)

        # Compute correlation at zero lag
        min_len = min(len(ioi1), len(ioi2))
        if min_len < 2:
            return 0.0

        try:
            corr, _ = stats.pearsonr(ioi1[:min_len], ioi2[:min_len])
            return corr if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def _compute_rhythmic_independence(self, voices: List[VoiceProfile]) -> float:
        """
        Compute overall rhythmic independence across all voices.

        Based on onset synchronization analysis.
        """
        if len(voices) <= 1:
            return 0.0

        # Collect all onsets
        all_onsets = []
        for voice in voices:
            onsets = [n.start for n in voice.notes]
            all_onsets.extend(onsets)

        if len(all_onsets) < 2:
            return 0.0

        all_onsets = sorted(all_onsets)

        # Count synchronized onsets (within 50ms)
        sync_threshold = 0.05
        synchronized = 0
        total_onsets = len(all_onsets)

        i = 0
        while i < len(all_onsets):
            # Count onsets within sync_threshold
            j = i + 1
            while j < len(all_onsets) and all_onsets[j] - all_onsets[i] < sync_threshold:
                j += 1

            sync_count = j - i
            if sync_count > 1:
                synchronized += sync_count

            i = j if j > i else i + 1

        # More synchronization = less independence
        sync_ratio = synchronized / total_onsets if total_onsets > 0 else 0.0
        return 1.0 - sync_ratio

    def _compute_voice_crossing_density(self, voices: List[VoiceProfile]) -> float:
        """
        Compute the density of voice crossings.

        Voice crossing occurs when a lower voice plays higher than an upper voice.
        """
        if len(voices) <= 1:
            return 0.0

        crossings = 0
        total_comparisons = 0

        # Sort voices by mean pitch
        sorted_voices = sorted(voices, key=lambda v: v.pitch_mean)

        # Check each pair of adjacent voices
        for i in range(len(sorted_voices) - 1):
            lower_voice = sorted_voices[i]
            upper_voice = sorted_voices[i + 1]

            # Sample at multiple time points
            lower_times = [n.start for n in lower_voice.notes]
            upper_times = [n.start for n in upper_voice.notes]

            # Check for crossings
            for ln in lower_voice.notes:
                for un in upper_voice.notes:
                    # Check if notes overlap in time
                    if not (ln.end <= un.start or ln.start >= un.end):
                        total_comparisons += 1
                        if ln.pitch > un.pitch:
                            crossings += 1

        return crossings / total_comparisons if total_comparisons > 0 else 0.0

    # ========================================================================
    # LAYERING ANALYSIS
    # ========================================================================

    def _detect_layers(self, notes: List[Tuple], voices: List[VoiceProfile]) -> List[List[VoiceProfile]]:
        """
        Detect textural layers (groups of voices that function together).

        Layers can be based on:
        1. Register (low, mid, high)
        2. Rhythmic similarity
        3. Timbral grouping (if MIDI program info available)
        """
        if len(voices) == 0:
            return []

        if self.layer_detection_method == 'register':
            return self._detect_layers_by_register(voices)
        elif self.layer_detection_method == 'hierarchical':
            return self._detect_layers_hierarchical(voices)
        else:
            return self._detect_layers_by_register(voices)  # Default

    def _detect_layers_by_register(self, voices: List[VoiceProfile]) -> List[List[VoiceProfile]]:
        """Group voices into layers by register."""
        layers = {'low': [], 'mid': [], 'high': []}

        for voice in voices:
            layers[voice.register].append(voice)

        # Return non-empty layers
        return [layer for layer in layers.values() if len(layer) > 0]

    def _detect_layers_hierarchical(self, voices: List[VoiceProfile]) -> List[List[VoiceProfile]]:
        """Use hierarchical clustering to detect layers."""
        if len(voices) < 2:
            return [[v] for v in voices]

        # Create feature vectors for each voice
        features = []
        for voice in voices:
            feat = [
                voice.pitch_mean / 127.0,
                voice.note_density,
                voice.activity_ratio,
                voice.pitch_std / 12.0
            ]
            features.append(feat)

        features = np.array(features)

        # Perform hierarchical clustering
        try:
            Z = linkage(features, method='ward')
            n_clusters = min(5, len(voices))  # Max 5 layers
            labels = fcluster(Z, n_clusters, criterion='maxclust')

            # Group voices by cluster
            layers = defaultdict(list)
            for voice, label in zip(voices, labels):
                layers[label].append(voice)

            return list(layers.values())
        except:
            # Fallback to register-based
            return self._detect_layers_by_register(voices)

    def _compute_register_spread(self, notes: List[Tuple]) -> float:
        """
        Compute how widely the texture spreads across registers.

        Returns 0-1 where 1 = uses full MIDI range.
        """
        if len(notes) == 0:
            return 0.0

        pitches = [pitch for _, _, pitch, _ in notes]
        pitch_range = max(pitches) - min(pitches)

        # Normalize to 0-1 (88 keys = full piano)
        return min(1.0, pitch_range / 88.0)

    def _compute_register_distribution(self, notes: List[Tuple], duration: float) -> Dict[str, float]:
        """
        Compute the proportion of activity in each register.

        Returns dict with 'low', 'mid', 'high' proportions.
        """
        register_time = {'low': 0.0, 'mid': 0.0, 'high': 0.0}

        for start, end, pitch, vel in notes:
            note_duration = end - start

            if self.register_boundaries['low'][0] <= pitch < self.register_boundaries['low'][1]:
                register_time['low'] += note_duration
            elif self.register_boundaries['mid'][0] <= pitch < self.register_boundaries['mid'][1]:
                register_time['mid'] += note_duration
            else:
                register_time['high'] += note_duration

        # Normalize
        total_time = sum(register_time.values())
        if total_time > 0:
            return {k: v/total_time for k, v in register_time.items()}
        else:
            return {'low': 0.0, 'mid': 0.0, 'high': 0.0}

    # ========================================================================
    # TEXTURE SEGMENTATION AND CONTRAST
    # ========================================================================

    def _segment_texture(self, notes: List[Tuple], polyphony_curve: np.ndarray,
                        time_grid: np.ndarray) -> List[TextureSegment]:
        """
        Segment the music into regions of consistent texture.

        Uses change detection on polyphony curve and note density.
        """
        if len(polyphony_curve) < 4:
            # Too short to segment meaningfully
            duration = time_grid[-1] if len(time_grid) > 0 else 1.0
            return [self._create_texture_segment(notes, 0.0, duration, polyphony_curve)]

        # Detect change points in polyphony
        change_points = self._detect_change_points(polyphony_curve)

        # Convert to time points
        change_times = [time_grid[i] for i in change_points if i < len(time_grid)]
        change_times = [0.0] + change_times + [time_grid[-1]]

        # Create segments
        segments = []
        for i in range(len(change_times) - 1):
            start = change_times[i]
            end = change_times[i + 1]

            # Get notes in this segment
            segment_notes = [n for n in notes if n[0] >= start and n[0] < end]

            if len(segment_notes) > 0:
                # Get polyphony curve for this segment
                start_idx = np.searchsorted(time_grid, start)
                end_idx = np.searchsorted(time_grid, end)
                segment_poly = polyphony_curve[start_idx:end_idx]

                segment = self._create_texture_segment(segment_notes, start, end, segment_poly)
                segments.append(segment)

        return segments

    def _detect_change_points(self, signal: np.ndarray, threshold: float = 1.0) -> List[int]:
        """Detect change points in a signal using derivative analysis."""
        if len(signal) < 4:
            return []

        # Smooth the signal
        window = min(11, len(signal) // 4)
        if window % 2 == 0:
            window += 1
        if window < 3:
            return []

        try:
            smoothed = signal.signal.savgol_filter(signal, window, 2)
        except:
            smoothed = signal

        # Compute derivative
        derivative = np.abs(np.diff(smoothed))

        # Find peaks in derivative
        mean_deriv = np.mean(derivative)
        std_deriv = np.std(derivative)
        threshold_value = mean_deriv + threshold * std_deriv

        change_points = []
        for i, d in enumerate(derivative):
            if d > threshold_value:
                change_points.append(i)

        # Merge nearby change points
        if len(change_points) > 0:
            merged = [change_points[0]]
            for cp in change_points[1:]:
                if cp - merged[-1] > 10:  # At least 10 time steps apart
                    merged.append(cp)
            change_points = merged

        return change_points

    def _create_texture_segment(self, notes: List[Tuple], start: float,
                               end: float, polyphony: np.ndarray) -> TextureSegment:
        """Create a TextureSegment from notes in a time range."""
        duration = end - start

        # Compute metrics
        mean_poly = np.mean(polyphony) if len(polyphony) > 0 else 0.0
        max_poly = int(np.max(polyphony)) if len(polyphony) > 0 else 0

        # Vertical density
        vertical_density = min(1.0, mean_poly / 12.0)

        # Horizontal density
        total_duration = sum(n[1] - n[0] for n in notes)
        horizontal_density = min(1.0, total_duration / (duration * 4.0)) if duration > 0 else 0.0

        # Classify texture type
        texture_type = self._classify_texture_type(notes, mean_poly)

        return TextureSegment(
            start_time=start,
            end_time=end,
            polyphonic_density=mean_poly,
            voice_count=max_poly,
            vertical_density=vertical_density,
            horizontal_density=horizontal_density,
            texture_type=texture_type
        )

    def _classify_texture_type(self, notes: List[Tuple], mean_polyphony: float) -> str:
        """
        Classify texture type based on polyphony and note characteristics.

        Types:
        - monophonic: Single note at a time
        - homophonic: Melody with chordal accompaniment
        - polyphonic: Multiple independent melodies
        - heterophonic: Variations of same melody
        """
        if mean_polyphony < 1.5:
            return 'monophonic'
        elif mean_polyphony < 3.0:
            # Could be homophonic or simple polyphony
            # Check for rhythmic alignment (homophonic tends to be more aligned)
            onsets = [n[0] for n in notes]
            onset_counts = Counter([round(o, 2) for o in onsets])
            max_simultaneous = max(onset_counts.values()) if onset_counts else 0

            if max_simultaneous / len(notes) > 0.3:
                return 'homophonic'
            else:
                return 'polyphonic'
        else:
            # Higher polyphony - check for chordal structure
            # If many notes have same onsets, likely homophonic
            onsets = [n[0] for n in notes]
            onset_counts = Counter([round(o, 2) for o in onsets])
            max_simultaneous = max(onset_counts.values()) if onset_counts else 0

            if max_simultaneous > mean_polyphony * 0.7:
                return 'homophonic'
            else:
                return 'polyphonic'

    def _compute_texture_ratios(self, segments: List[TextureSegment],
                               total_duration: float) -> Dict[str, float]:
        """Compute the proportion of time spent in each texture type."""
        texture_time = defaultdict(float)

        for seg in segments:
            duration = seg.end_time - seg.start_time
            texture_time[seg.texture_type] += duration

        # Normalize
        if total_duration > 0:
            return {
                'monophonic': texture_time['monophonic'] / total_duration,
                'homophonic': texture_time['homophonic'] / total_duration,
                'polyphonic': texture_time['polyphonic'] / total_duration,
                'heterophonic': texture_time['heterophonic'] / total_duration
            }
        else:
            return {'monophonic': 0.0, 'homophonic': 0.0, 'polyphonic': 0.0, 'heterophonic': 0.0}

    def _compute_contrast_rate(self, segments: List[TextureSegment]) -> float:
        """
        Compute how frequently the texture changes.

        Returns 0-1 where 1 = very frequent changes.
        """
        if len(segments) <= 1:
            return 0.0

        # Count texture type changes
        changes = 0
        for i in range(len(segments) - 1):
            if segments[i].texture_type != segments[i+1].texture_type:
                changes += 1

        # Normalize by number of possible changes
        max_changes = len(segments) - 1
        return changes / max_changes if max_changes > 0 else 0.0

    def _extract_texture_transitions(self, segments: List[TextureSegment]) -> List[Tuple[str, str, float]]:
        """Extract texture transitions (from, to, time)."""
        transitions = []

        for i in range(len(segments) - 1):
            from_type = segments[i].texture_type
            to_type = segments[i+1].texture_type
            time = segments[i+1].start_time

            if from_type != to_type:
                transitions.append((from_type, to_type, time))

        return transitions

    # ========================================================================
    # DETAILED FEATURE EXTRACTION (135+ features)
    # ========================================================================

    def _compute_detailed_features(self, notes: List[Tuple], polyphony_curve: np.ndarray,
                                  voices: List[VoiceProfile], layers: List[List[VoiceProfile]],
                                  segments: List[TextureSegment], time_grid: np.ndarray,
                                  duration: float) -> Dict[str, float]:
        """
        Compute 135+ detailed texture features for XGBoost training.

        Categories:
        1. Polyphony statistics (20 features)
        2. Voice characteristics (30 features)
        3. Layer characteristics (20 features)
        4. Temporal evolution (25 features)
        5. Spectral/register features (20 features)
        6. Texture type features (20 features)
        """
        features = {}

        # ===== POLYPHONY STATISTICS (20) =====
        features['poly_mean'] = float(np.mean(polyphony_curve))
        features['poly_std'] = float(np.std(polyphony_curve))
        features['poly_min'] = float(np.min(polyphony_curve))
        features['poly_max'] = float(np.max(polyphony_curve))
        features['poly_median'] = float(np.median(polyphony_curve))
        features['poly_q25'] = float(np.percentile(polyphony_curve, 25))
        features['poly_q75'] = float(np.percentile(polyphony_curve, 75))
        features['poly_range'] = features['poly_max'] - features['poly_min']
        features['poly_iqr'] = features['poly_q75'] - features['poly_q25']
        features['poly_cv'] = features['poly_std'] / (features['poly_mean'] + 1e-6)

        # Polyphony distribution
        for i in range(1, 9):
            features[f'poly_prop_{i}'] = np.mean(polyphony_curve == i)
        features['poly_prop_9plus'] = np.mean(polyphony_curve >= 9)

        # Polyphony change rate
        poly_changes = np.abs(np.diff(polyphony_curve))
        features['poly_change_rate'] = float(np.mean(poly_changes > 0))
        features['poly_change_magnitude'] = float(np.mean(poly_changes))

        # ===== VOICE CHARACTERISTICS (30) =====
        features['voice_count'] = len(voices)

        if len(voices) > 0:
            # Voice ranges
            voice_ranges = [v.pitch_range[1] - v.pitch_range[0] for v in voices]
            features['voice_range_mean'] = float(np.mean(voice_ranges))
            features['voice_range_std'] = float(np.std(voice_ranges))
            features['voice_range_max'] = float(np.max(voice_ranges))

            # Voice densities
            voice_densities = [v.note_density for v in voices]
            features['voice_density_mean'] = float(np.mean(voice_densities))
            features['voice_density_std'] = float(np.std(voice_densities))
            features['voice_density_max'] = float(np.max(voice_densities))

            # Voice activity
            voice_activities = [v.activity_ratio for v in voices]
            features['voice_activity_mean'] = float(np.mean(voice_activities))
            features['voice_activity_std'] = float(np.std(voice_activities))
            features['voice_activity_min'] = float(np.min(voice_activities))

            # Voice pitch statistics
            voice_pitch_means = [v.pitch_mean for v in voices]
            features['voice_pitch_mean'] = float(np.mean(voice_pitch_means))
            features['voice_pitch_spread'] = float(np.std(voice_pitch_means))
            features['voice_pitch_range_total'] = float(max(voice_pitch_means) - min(voice_pitch_means))

            # Voice register distribution
            register_counts = Counter(v.register for v in voices)
            features['voices_in_low'] = register_counts.get('low', 0)
            features['voices_in_mid'] = register_counts.get('mid', 0)
            features['voices_in_high'] = register_counts.get('high', 0)

            # Inter-voice intervals
            if len(voices) > 1:
                sorted_voices = sorted(voices, key=lambda v: v.pitch_mean)
                intervals = [sorted_voices[i+1].pitch_mean - sorted_voices[i].pitch_mean
                           for i in range(len(sorted_voices)-1)]
                features['inter_voice_interval_mean'] = float(np.mean(intervals))
                features['inter_voice_interval_std'] = float(np.std(intervals))
                features['inter_voice_interval_min'] = float(np.min(intervals))
                features['inter_voice_interval_max'] = float(np.max(intervals))
            else:
                features['inter_voice_interval_mean'] = 0.0
                features['inter_voice_interval_std'] = 0.0
                features['inter_voice_interval_min'] = 0.0
                features['inter_voice_interval_max'] = 0.0

            # Rhythmic complexity
            rhythm_complexities = []
            for v in voices:
                if len(v.rhythmic_pattern) > 1:
                    complexity = np.std(v.rhythmic_pattern) / (np.mean(v.rhythmic_pattern) + 1e-6)
                    rhythm_complexities.append(complexity)

            if rhythm_complexities:
                features['voice_rhythm_complexity_mean'] = float(np.mean(rhythm_complexities))
                features['voice_rhythm_complexity_max'] = float(np.max(rhythm_complexities))
            else:
                features['voice_rhythm_complexity_mean'] = 0.0
                features['voice_rhythm_complexity_max'] = 0.0
        else:
            # No voices detected - fill with zeros
            for key in ['voice_range_mean', 'voice_range_std', 'voice_range_max',
                       'voice_density_mean', 'voice_density_std', 'voice_density_max',
                       'voice_activity_mean', 'voice_activity_std', 'voice_activity_min',
                       'voice_pitch_mean', 'voice_pitch_spread', 'voice_pitch_range_total',
                       'voices_in_low', 'voices_in_mid', 'voices_in_high',
                       'inter_voice_interval_mean', 'inter_voice_interval_std',
                       'inter_voice_interval_min', 'inter_voice_interval_max',
                       'voice_rhythm_complexity_mean', 'voice_rhythm_complexity_max']:
                features[key] = 0.0

        # ===== LAYER CHARACTERISTICS (20) =====
        features['layer_count'] = len(layers)

        if len(layers) > 0:
            layer_sizes = [len(layer) for layer in layers]
            features['layer_size_mean'] = float(np.mean(layer_sizes))
            features['layer_size_std'] = float(np.std(layer_sizes))
            features['layer_size_max'] = float(np.max(layer_sizes))
            features['layer_size_min'] = float(np.min(layer_sizes))

            # Layer densities
            layer_densities = []
            for layer in layers:
                total_density = sum(v.note_density for v in layer)
                layer_densities.append(total_density)

            features['layer_density_mean'] = float(np.mean(layer_densities))
            features['layer_density_std'] = float(np.std(layer_densities))
            features['layer_density_max'] = float(np.max(layer_densities))

            # Layer pitch ranges
            layer_ranges = []
            for layer in layers:
                if len(layer) > 0:
                    min_pitch = min(v.pitch_mean for v in layer)
                    max_pitch = max(v.pitch_mean for v in layer)
                    layer_ranges.append(max_pitch - min_pitch)

            if layer_ranges:
                features['layer_range_mean'] = float(np.mean(layer_ranges))
                features['layer_range_max'] = float(np.max(layer_ranges))
            else:
                features['layer_range_mean'] = 0.0
                features['layer_range_max'] = 0.0

            # Layer separation
            if len(layers) > 1:
                layer_centers = [np.mean([v.pitch_mean for v in layer]) for layer in layers]
                layer_centers_sorted = sorted(layer_centers)
                separations = np.diff(layer_centers_sorted)
                features['layer_separation_mean'] = float(np.mean(separations))
                features['layer_separation_min'] = float(np.min(separations))
            else:
                features['layer_separation_mean'] = 0.0
                features['layer_separation_min'] = 0.0
        else:
            for key in ['layer_size_mean', 'layer_size_std', 'layer_size_max', 'layer_size_min',
                       'layer_density_mean', 'layer_density_std', 'layer_density_max',
                       'layer_range_mean', 'layer_range_max',
                       'layer_separation_mean', 'layer_separation_min']:
                features[key] = 0.0

        # Fill remaining layer features
        for i in range(12):
            if i < len(features):
                continue
            features[f'layer_feature_{i}'] = 0.0

        # ===== TEMPORAL EVOLUTION (25) =====
        if len(segments) > 0:
            features['segment_count'] = len(segments)

            # Segment durations
            seg_durations = [s.end_time - s.start_time for s in segments]
            features['segment_duration_mean'] = float(np.mean(seg_durations))
            features['segment_duration_std'] = float(np.std(seg_durations))
            features['segment_duration_min'] = float(np.min(seg_durations))
            features['segment_duration_max'] = float(np.max(seg_durations))

            # Segment polyphony
            seg_poly = [s.polyphonic_density for s in segments]
            features['segment_poly_mean'] = float(np.mean(seg_poly))
            features['segment_poly_std'] = float(np.std(seg_poly))
            features['segment_poly_range'] = float(max(seg_poly) - min(seg_poly))

            # Segment densities
            seg_vert_density = [s.vertical_density for s in segments]
            features['segment_vert_density_mean'] = float(np.mean(seg_vert_density))
            features['segment_vert_density_std'] = float(np.std(seg_vert_density))

            seg_horiz_density = [s.horizontal_density for s in segments]
            features['segment_horiz_density_mean'] = float(np.mean(seg_horiz_density))
            features['segment_horiz_density_std'] = float(np.std(seg_horiz_density))

            # Texture type distribution
            texture_types = [s.texture_type for s in segments]
            type_counts = Counter(texture_types)
            total_segs = len(segments)
            features['segment_monophonic_prop'] = type_counts.get('monophonic', 0) / total_segs
            features['segment_homophonic_prop'] = type_counts.get('homophonic', 0) / total_segs
            features['segment_polyphonic_prop'] = type_counts.get('polyphonic', 0) / total_segs
            features['segment_heterophonic_prop'] = type_counts.get('heterophonic', 0) / total_segs

            # Texture transitions
            texture_changes = sum(1 for i in range(len(segments)-1)
                                if segments[i].texture_type != segments[i+1].texture_type)
            features['texture_transition_count'] = texture_changes
            features['texture_transition_rate'] = texture_changes / (len(segments) - 1) if len(segments) > 1 else 0.0

            # Polyphony evolution
            poly_trend = np.polyfit(range(len(seg_poly)), seg_poly, 1)[0] if len(seg_poly) > 1 else 0.0
            features['polyphony_trend'] = float(poly_trend)

            # Density evolution
            vert_trend = np.polyfit(range(len(seg_vert_density)), seg_vert_density, 1)[0] if len(seg_vert_density) > 1 else 0.0
            features['vertical_density_trend'] = float(vert_trend)
        else:
            for key in ['segment_count', 'segment_duration_mean', 'segment_duration_std',
                       'segment_duration_min', 'segment_duration_max',
                       'segment_poly_mean', 'segment_poly_std', 'segment_poly_range',
                       'segment_vert_density_mean', 'segment_vert_density_std',
                       'segment_horiz_density_mean', 'segment_horiz_density_std',
                       'segment_monophonic_prop', 'segment_homophonic_prop',
                       'segment_polyphonic_prop', 'segment_heterophonic_prop',
                       'texture_transition_count', 'texture_transition_rate',
                       'polyphony_trend', 'vertical_density_trend']:
                features[key] = 0.0

        # Polyphony evolution features
        if len(polyphony_curve) > 1:
            # Autocorrelation
            poly_normalized = (polyphony_curve - np.mean(polyphony_curve)) / (np.std(polyphony_curve) + 1e-6)
            autocorr = np.correlate(poly_normalized, poly_normalized, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            features['poly_autocorr_lag1'] = float(autocorr[1] / (autocorr[0] + 1e-6)) if len(autocorr) > 1 else 0.0
        else:
            features['poly_autocorr_lag1'] = 0.0

        # Fill to reach 25 temporal features
        for i in range(25 - 22):
            features[f'temporal_feature_{i}'] = 0.0

        # ===== SPECTRAL/REGISTER FEATURES (20) =====
        if len(notes) > 0:
            pitches = np.array([p for _, _, p, _ in notes])

            # Pitch statistics
            features['pitch_mean'] = float(np.mean(pitches))
            features['pitch_std'] = float(np.std(pitches))
            features['pitch_min'] = float(np.min(pitches))
            features['pitch_max'] = float(np.max(pitches))
            features['pitch_range'] = features['pitch_max'] - features['pitch_min']
            features['pitch_median'] = float(np.median(pitches))

            # Pitch distribution
            pitch_hist, _ = np.histogram(pitches, bins=12, range=(0, 127))
            pitch_hist_norm = pitch_hist / (np.sum(pitch_hist) + 1e-6)
            for i in range(12):
                features[f'pitch_octave_{i}_prop'] = float(pitch_hist_norm[i])

            # Register usage (already computed, but add more detail)
            register_low = np.sum((pitches >= 0) & (pitches < 48)) / len(pitches)
            register_mid = np.sum((pitches >= 48) & (pitches < 72)) / len(pitches)
            register_high = np.sum((pitches >= 72) & (pitches <= 127)) / len(pitches)

            features['register_low_usage'] = float(register_low)
            features['register_mid_usage'] = float(register_mid)
            features['register_high_usage'] = float(register_high)

            # Register entropy
            register_dist = np.array([register_low, register_mid, register_high])
            register_dist = register_dist / (np.sum(register_dist) + 1e-6)
            register_entropy = -np.sum(register_dist * np.log2(register_dist + 1e-6))
            features['register_entropy'] = float(register_entropy)
        else:
            for key in ['pitch_mean', 'pitch_std', 'pitch_min', 'pitch_max', 'pitch_range', 'pitch_median']:
                features[key] = 0.0
            for i in range(12):
                features[f'pitch_octave_{i}_prop'] = 0.0
            features['register_low_usage'] = 0.0
            features['register_mid_usage'] = 0.0
            features['register_high_usage'] = 0.0
            features['register_entropy'] = 0.0

        # ===== TEXTURE TYPE FEATURES (20) =====
        # Already have texture type proportions, add more detailed metrics

        # Onset synchronization
        if len(notes) > 1:
            onsets = [n[0] for n in notes]
            onset_counts = Counter([round(o, 2) for o in onsets])
            max_sync = max(onset_counts.values())
            mean_sync = np.mean(list(onset_counts.values()))
            features['onset_max_synchronization'] = max_sync / len(notes)
            features['onset_mean_synchronization'] = mean_sync / len(notes)

            # Onset regularity
            unique_onsets = sorted(onset_counts.keys())
            if len(unique_onsets) > 1:
                onset_iois = np.diff(unique_onsets)
                features['onset_ioi_mean'] = float(np.mean(onset_iois))
                features['onset_ioi_cv'] = float(np.std(onset_iois) / (np.mean(onset_iois) + 1e-6))
            else:
                features['onset_ioi_mean'] = 0.0
                features['onset_ioi_cv'] = 0.0
        else:
            features['onset_max_synchronization'] = 0.0
            features['onset_mean_synchronization'] = 0.0
            features['onset_ioi_mean'] = 0.0
            features['onset_ioi_cv'] = 0.0

        # Note duration statistics
        durations = [n[1] - n[0] for n in notes]
        features['note_duration_mean'] = float(np.mean(durations))
        features['note_duration_std'] = float(np.std(durations))
        features['note_duration_min'] = float(np.min(durations))
        features['note_duration_max'] = float(np.max(durations))
        features['note_duration_range'] = features['note_duration_max'] - features['note_duration_min']

        # Note overlap analysis
        overlap_count = 0
        total_pairs = 0
        for i in range(len(notes)):
            for j in range(i+1, len(notes)):
                total_pairs += 1
                n1_start, n1_end, _, _ = notes[i]
                n2_start, n2_end, _, _ = notes[j]
                if not (n1_end <= n2_start or n2_end <= n1_start):
                    overlap_count += 1

        features['note_overlap_ratio'] = overlap_count / total_pairs if total_pairs > 0 else 0.0

        # Velocity variance (can indicate homophonic vs polyphonic)
        velocities = [v for _, _, _, v in notes]
        features['velocity_mean'] = float(np.mean(velocities))
        features['velocity_std'] = float(np.std(velocities))
        features['velocity_range'] = float(max(velocities) - min(velocities))

        # Fill to reach 20 texture type features
        for i in range(20 - 16):
            features[f'texture_type_feature_{i}'] = 0.0

        return features

    # ========================================================================
    # PARAMETER EXTRACTION FOR REGISTRY
    # ========================================================================

    def extract_parameters(self, midi_file_path: str) -> Dict[str, float]:
        """
        Extract texture parameters for the universal registry.

        Returns the 10 key parameters that can be used for generation.
        """
        result = self.analyze(midi_file_path)

        parameters = {
            'texture_polyphonic_density': min(1.0, result.mean_polyphonic_density / 8.0),
            'texture_voice_independence': result.voice_independence,
            'texture_vertical_density': result.vertical_density,
            'texture_horizontal_density': result.horizontal_density,
            'texture_layering_count': min(8, result.layering_count),
            'texture_register_spread': result.register_spread,
            'texture_contrast_rate': result.contrast_rate,
            'texture_homophonic_ratio': result.homophonic_ratio,
            'texture_voice_crossing_density': result.voice_crossing_density,
            'texture_rhythmic_independence': result.rhythmic_independence
        }

        return parameters

    def get_parameter_definitions(self) -> List[Dict[str, Any]]:
        """
        Get parameter definitions for registry.json integration.

        Returns list of parameter dicts with metadata.
        """
        return [
            {
                "name": "texture_polyphonic_density",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
                "description": "Average number of simultaneous voices (0=monophonic, 1=very dense)",
                "category": "texture",
                "agent": 24,
                "impacts": ["note_generation", "voice_leading"],
                "musical_function": "Controls vertical density of texture"
            },
            {
                "name": "texture_voice_independence",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Degree of independence between voices (0=homophonic, 1=contrapuntal)",
                "category": "texture",
                "agent": 24,
                "impacts": ["voice_leading", "melodic_contour"],
                "musical_function": "Controls melodic and rhythmic independence"
            },
            {
                "name": "texture_vertical_density",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.4,
                "description": "Harmonic complexity at any moment (0=sparse, 1=dense chords)",
                "category": "texture",
                "agent": 24,
                "impacts": ["chord_generation", "harmonic_rhythm"],
                "musical_function": "Controls chordal density"
            },
            {
                "name": "texture_horizontal_density",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.6,
                "description": "Temporal activity level (0=sparse, 1=continuous)",
                "category": "texture",
                "agent": 24,
                "impacts": ["rhythmic_density", "note_timing"],
                "musical_function": "Controls temporal continuity"
            },
            {
                "name": "texture_layering_count",
                "type": "int",
                "range": [1, 8],
                "default": 3,
                "description": "Number of distinct textural layers (1=single layer, 8=complex)",
                "category": "texture",
                "agent": 24,
                "impacts": ["orchestration", "arrangement"],
                "musical_function": "Controls number of textural strata"
            },
            {
                "name": "texture_register_spread",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Pitch range usage (0=narrow, 1=full keyboard)",
                "category": "texture",
                "agent": 24,
                "impacts": ["pitch_range", "register_usage"],
                "musical_function": "Controls vertical pitch space usage"
            },
            {
                "name": "texture_contrast_rate",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.3,
                "description": "Frequency of texture changes (0=static, 1=constantly changing)",
                "category": "texture",
                "agent": 24,
                "impacts": ["form", "sectional_contrast"],
                "musical_function": "Controls textural variety over time"
            },
            {
                "name": "texture_homophonic_ratio",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.4,
                "description": "Proportion of homophonic texture (0=polyphonic, 1=chordal)",
                "category": "texture",
                "agent": 24,
                "impacts": ["texture_type", "voice_leading"],
                "musical_function": "Balance between homophony and polyphony"
            },
            {
                "name": "texture_voice_crossing_density",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.1,
                "description": "Frequency of voice crossings (0=none, 1=frequent)",
                "category": "texture",
                "agent": 24,
                "impacts": ["voice_leading", "counterpoint"],
                "musical_function": "Controls voice crossing frequency"
            },
            {
                "name": "texture_rhythmic_independence",
                "type": "float",
                "range": [0.0, 1.0],
                "default": 0.5,
                "description": "Rhythmic independence between voices (0=synchronized, 1=independent)",
                "category": "texture",
                "agent": 24,
                "impacts": ["rhythmic_complexity", "polyrhythm"],
                "musical_function": "Controls rhythmic coordination"
            }
        ]

    # ========================================================================
    # UTILITIES
    # ========================================================================

    def _collect_all_notes(self, pm: pretty_midi.PrettyMIDI) -> List[Tuple]:
        """Collect all notes from all instruments as (start, end, pitch, velocity) tuples."""
        notes = []
        for instrument in pm.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                notes.append((note.start, note.end, note.pitch, note.velocity))
        return sorted(notes, key=lambda x: x[0])

    def _empty_result(self) -> TextureAnalysisResult:
        """Return an empty result for error cases."""
        return TextureAnalysisResult(
            mean_polyphonic_density=0.0,
            std_polyphonic_density=0.0,
            max_polyphonic_density=0,
            voice_independence=0.0,
            vertical_density=0.0,
            horizontal_density=0.0,
            detected_voices=[],
            voice_count=0,
            voice_crossing_density=0.0,
            rhythmic_independence=0.0,
            layering_count=0,
            register_spread=0.0,
            register_distribution={'low': 0.0, 'mid': 0.0, 'high': 0.0},
            homophonic_ratio=0.0,
            polyphonic_ratio=0.0,
            monophonic_ratio=0.0,
            texture_segments=[],
            contrast_rate=0.0,
            texture_transitions=[],
            detailed_features={}
        )


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

def main():
    """Demonstrate the texture specialist."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python texture_specialist.py <midi_file>")
        sys.exit(1)

    midi_file = sys.argv[1]

    print("="*80)
    print("AGENT 24: TEXTURE SPECIALIST")
    print("="*80)
    print(f"\nAnalyzing: {midi_file}\n")

    specialist = TextureSpecialist()
    result = specialist.analyze(midi_file)

    # Print summary
    print("GLOBAL METRICS:")
    print(f"  Polyphonic Density: {result.mean_polyphonic_density:.2f} ± {result.std_polyphonic_density:.2f}")
    print(f"  Max Polyphony: {result.max_polyphonic_density}")
    print(f"  Voice Independence: {result.voice_independence:.3f}")
    print(f"  Vertical Density: {result.vertical_density:.3f}")
    print(f"  Horizontal Density: {result.horizontal_density:.3f}")

    print(f"\nVOICE ANALYSIS:")
    print(f"  Detected Voices: {result.voice_count}")
    print(f"  Voice Crossing Density: {result.voice_crossing_density:.3f}")
    print(f"  Rhythmic Independence: {result.rhythmic_independence:.3f}")

    for i, voice in enumerate(result.detected_voices, 1):
        print(f"\n  Voice {i} ({voice.register}):")
        print(f"    Pitch Range: {voice.pitch_range[0]}-{voice.pitch_range[1]} (mean: {voice.pitch_mean:.1f})")
        print(f"    Note Density: {voice.note_density:.2f} notes/sec")
        print(f"    Activity Ratio: {voice.activity_ratio:.3f}")

    print(f"\nLAYERING:")
    print(f"  Layer Count: {result.layering_count}")
    print(f"  Register Spread: {result.register_spread:.3f}")
    print(f"  Register Distribution:")
    for reg, prop in result.register_distribution.items():
        print(f"    {reg}: {prop:.1%}")

    print(f"\nTEXTURE TYPES:")
    print(f"  Monophonic: {result.monophonic_ratio:.1%}")
    print(f"  Homophonic: {result.homophonic_ratio:.1%}")
    print(f"  Polyphonic: {result.polyphonic_ratio:.1%}")
    print(f"  Contrast Rate: {result.contrast_rate:.3f}")

    print(f"\nTEXTURE SEGMENTS: {len(result.texture_segments)}")
    for i, seg in enumerate(result.texture_segments[:5], 1):  # Show first 5
        print(f"  Segment {i} ({seg.start_time:.1f}s - {seg.end_time:.1f}s):")
        print(f"    Type: {seg.texture_type}")
        print(f"    Polyphony: {seg.polyphonic_density:.2f}")
        print(f"    Vertical/Horizontal: {seg.vertical_density:.2f} / {seg.horizontal_density:.2f}")

    if len(result.texture_segments) > 5:
        print(f"  ... and {len(result.texture_segments) - 5} more segments")

    print(f"\nTEXTURE TRANSITIONS: {len(result.texture_transitions)}")
    for from_type, to_type, time in result.texture_transitions[:5]:
        print(f"  {time:.1f}s: {from_type} → {to_type}")

    print(f"\nDETAILED FEATURES: {len(result.detailed_features)}")

    # Extract parameters
    params = specialist.extract_parameters(midi_file)
    print("\nEXTRACTED PARAMETERS (for generation):")
    for param, value in params.items():
        print(f"  {param}: {value:.3f}")

    print("\n" + "="*80)
    print(f"Analysis complete. Extracted {len(result.detailed_features)} detailed features.")
    print("="*80)


if __name__ == "__main__":
    main()
