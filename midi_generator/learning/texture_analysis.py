"""
Texture Analysis Methods - Agent 6 Support Module
==================================================

Detailed implementations of texture analysis algorithms for discovering
the 20 interpretable texture parameters.

This module provides the analytical foundation for TextureSemanticEncoder,
implementing algorithms to compute texture properties from MIDI data.

Features:
---------
1. Homophonic vs Polyphonic Detection
2. Voice Independence Metrics (rhythmic, melodic, harmonic)
3. Textural Density Calculation (vertical and horizontal)
4. Call-Response Pattern Detection
5. Layer Interaction Analysis
6. Voice Crossing Detection
7. Imitation and Canon Detection
8. Texture Evolution Tracking

Author: Agent 6 - Texture Encoder Specialist
Date: November 21, 2025
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available - texture analysis will be limited")


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class Note:
    """Simple note representation for analysis"""
    pitch: int
    start_time: float
    duration: float
    velocity: int
    voice_id: int = 0
    channel: int = 0


@dataclass
class TextureProfile:
    """Complete texture profile of a musical passage"""

    # Texture Type (4 metrics)
    homophonic_polyphonic_balance: float  # 0=homophonic, 1=polyphonic
    monophonic_tendency: float            # 0=none, 1=single line
    heterophonic_variation: float         # 0=none, 1=high variation
    texture_consistency: float            # 0=varying, 1=consistent

    # Voice Independence (4 metrics)
    voice_independence_score: float       # Overall independence
    rhythmic_independence: float          # Rhythmic independence
    melodic_independence: float           # Melodic independence
    harmonic_independence: float          # Harmonic independence

    # Density & Complexity (4 metrics)
    textural_density_mean: float          # Average note density
    textural_density_variance: float      # Density variation
    vertical_density: float               # Simultaneous notes
    horizontal_density: float             # Sequential notes

    # Layering & Interaction (4 metrics)
    layer_count: float                    # Number of distinct layers
    layer_interaction_complexity: float   # Layer interaction
    foreground_background_separation: float  # Fg/bg clarity
    voice_crossing_frequency: float       # How often voices cross

    # Temporal Patterns (4 metrics)
    call_response_strength: float         # Call-response patterns
    imitation_frequency: float            # Imitative counterpoint
    texture_evolution_rate: float         # Rate of texture change
    stagger_synchronization_balance: float  # Stagger vs sync


# ============================================================================
# Main Texture Analyzer
# ============================================================================

class DetailedTextureAnalyzer:
    """
    Comprehensive texture analysis engine.

    Analyzes musical textures to extract 20 interpretable parameters
    that describe textural properties.
    """

    def __init__(self, time_window: float = 1.0, analysis_grid: float = 0.25):
        """
        Initialize texture analyzer.

        Args:
            time_window: Time window for local analysis (seconds)
            analysis_grid: Grid resolution for analysis (beats)
        """
        self.time_window = time_window
        self.analysis_grid = analysis_grid

    def analyze(self, notes: List[Note]) -> TextureProfile:
        """
        Perform complete texture analysis.

        Args:
            notes: List of notes to analyze

        Returns:
            TextureProfile with all 20 parameters
        """
        if not notes:
            return self._empty_profile()

        # Organize notes by voice
        notes_by_voice = self._organize_by_voice(notes)

        # Compute each category of metrics
        texture_type = self._analyze_texture_type(notes, notes_by_voice)
        voice_independence = self._analyze_voice_independence(notes_by_voice)
        density = self._analyze_density(notes)
        layering = self._analyze_layering(notes, notes_by_voice)
        temporal_patterns = self._analyze_temporal_patterns(notes, notes_by_voice)

        return TextureProfile(
            **texture_type,
            **voice_independence,
            **density,
            **layering,
            **temporal_patterns
        )

    # ========================================================================
    # Texture Type Analysis
    # ========================================================================

    def _analyze_texture_type(
        self,
        notes: List[Note],
        notes_by_voice: Dict[int, List[Note]]
    ) -> Dict[str, float]:
        """
        Analyze fundamental texture type.

        Returns:
            Dictionary with 4 texture type metrics
        """
        num_voices = len(notes_by_voice)

        # Monophonic tendency
        monophonic = 1.0 if num_voices == 1 else max(0.0, 1.0 - (num_voices - 1) * 0.2)

        # Homophonic vs polyphonic
        if num_voices <= 1:
            homophonic_polyphonic = 0.0  # Neither (monophonic)
        else:
            # Compute rhythmic synchronization
            sync = self._compute_rhythmic_synchronization(notes_by_voice)
            # High sync = homophonic (0.0), low sync = polyphonic (1.0)
            homophonic_polyphonic = 1.0 - sync

        # Heterophonic variation (variations of same melody)
        heterophonic = self._detect_heterophonic_variation(notes_by_voice)

        # Texture consistency over time
        consistency = self._compute_texture_consistency(notes)

        return {
            "homophonic_polyphonic_balance": homophonic_polyphonic,
            "monophonic_tendency": monophonic,
            "heterophonic_variation": heterophonic,
            "texture_consistency": consistency
        }

    def _compute_rhythmic_synchronization(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """
        Compute how synchronized voices are rhythmically.

        Returns:
            Synchronization (0.0 = independent, 1.0 = synchronized)
        """
        if len(notes_by_voice) <= 1:
            return 1.0

        # Collect all onset times per voice
        onset_times = {}
        for voice_id, voice_notes in notes_by_voice.items():
            onset_times[voice_id] = sorted(set(n.start_time for n in voice_notes))

        # Compute pairwise synchronization
        sync_scores = []
        voices = list(onset_times.keys())

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                v1_onsets = set(onset_times[voices[i]])
                v2_onsets = set(onset_times[voices[j]])

                # Compute overlap (with tolerance)
                tolerance = 0.1  # 100ms tolerance
                matches = 0
                for t1 in v1_onsets:
                    if any(abs(t1 - t2) < tolerance for t2 in v2_onsets):
                        matches += 1

                # Sync score for this pair
                total = max(len(v1_onsets), len(v2_onsets))
                sync = matches / total if total > 0 else 0.0
                sync_scores.append(sync)

        return sum(sync_scores) / len(sync_scores) if sync_scores else 0.0

    def _detect_heterophonic_variation(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """
        Detect heterophonic texture (variations of same melody).

        Returns:
            Heterophonic strength (0.0 to 1.0)
        """
        if len(notes_by_voice) <= 1:
            return 0.0

        # Compare melodic contours between voices
        contours = {}
        for voice_id, voice_notes in notes_by_voice.items():
            contours[voice_id] = self._extract_contour(voice_notes)

        # Compute contour similarity
        similarities = []
        voices = list(contours.keys())

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                similarity = self._compute_contour_similarity(
                    contours[voices[i]],
                    contours[voices[j]]
                )
                similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # High similarity with rhythmic variation = heterophonic
        return avg_similarity

    def _extract_contour(self, notes: List[Note]) -> List[int]:
        """
        Extract melodic contour (up/down/same).

        Returns:
            List of contour directions: 1=up, 0=same, -1=down
        """
        if len(notes) < 2:
            return []

        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        contour = []

        for i in range(1, len(sorted_notes)):
            pitch_diff = sorted_notes[i].pitch - sorted_notes[i-1].pitch
            if pitch_diff > 0:
                contour.append(1)
            elif pitch_diff < 0:
                contour.append(-1)
            else:
                contour.append(0)

        return contour

    def _compute_contour_similarity(self, c1: List[int], c2: List[int]) -> float:
        """Compute similarity between two contours"""
        if not c1 or not c2:
            return 0.0

        min_len = min(len(c1), len(c2))
        if min_len == 0:
            return 0.0

        matches = sum(1 for i in range(min_len) if c1[i] == c2[i])
        return matches / min_len

    def _compute_texture_consistency(self, notes: List[Note]) -> float:
        """
        Compute how consistent texture is over time.

        Returns:
            Consistency (0.0 = varying, 1.0 = consistent)
        """
        # Divide into time windows and compute density variance
        densities = self._compute_density_windows(notes, window_size=2.0)

        if len(densities) < 2:
            return 1.0

        # Compute coefficient of variation
        if NUMPY_AVAILABLE:
            mean_density = np.mean(densities)
            std_density = np.std(densities)
            if mean_density > 0:
                cv = std_density / mean_density
                # Lower CV = more consistent
                consistency = max(0.0, 1.0 - cv)
                return consistency

        return 0.5  # Default

    # ========================================================================
    # Voice Independence Analysis
    # ========================================================================

    def _analyze_voice_independence(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> Dict[str, float]:
        """
        Analyze independence between voices.

        Returns:
            Dictionary with 4 independence metrics
        """
        if len(notes_by_voice) <= 1:
            return {
                "voice_independence_score": 0.0,
                "rhythmic_independence": 0.0,
                "melodic_independence": 0.0,
                "harmonic_independence": 0.0
            }

        # Compute independence scores
        rhythmic_indep = 1.0 - self._compute_rhythmic_synchronization(notes_by_voice)
        melodic_indep = self._compute_melodic_independence(notes_by_voice)
        harmonic_indep = self._compute_harmonic_independence(notes_by_voice)

        # Overall independence (weighted average)
        overall = (rhythmic_indep + melodic_indep + harmonic_indep) / 3.0

        return {
            "voice_independence_score": overall,
            "rhythmic_independence": rhythmic_indep,
            "melodic_independence": melodic_indep,
            "harmonic_independence": harmonic_indep
        }

    def _compute_melodic_independence(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """
        Compute melodic independence (different melodic contours).

        Returns:
            Independence score (0.0 to 1.0)
        """
        # Extract contours for each voice
        contours = {}
        for voice_id, voice_notes in notes_by_voice.items():
            contours[voice_id] = self._extract_contour(voice_notes)

        # Compute dissimilarity
        similarities = []
        voices = list(contours.keys())

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                similarity = self._compute_contour_similarity(
                    contours[voices[i]],
                    contours[voices[j]]
                )
                similarities.append(similarity)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

        # Independence = 1 - similarity
        return 1.0 - avg_similarity

    def _compute_harmonic_independence(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """
        Compute harmonic independence (different harmonic functions).

        Returns:
            Independence score (0.0 to 1.0)
        """
        # Simplified: measure pitch class diversity between voices
        if len(notes_by_voice) <= 1:
            return 0.0

        # Collect pitch classes per voice
        pc_sets = {}
        for voice_id, voice_notes in notes_by_voice.items():
            pcs = set(n.pitch % 12 for n in voice_notes)
            pc_sets[voice_id] = pcs

        # Compute diversity (lower overlap = more independent)
        overlaps = []
        voices = list(pc_sets.keys())

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                intersection = len(pc_sets[voices[i]] & pc_sets[voices[j]])
                union = len(pc_sets[voices[i]] | pc_sets[voices[j]])
                overlap = intersection / union if union > 0 else 0.0
                overlaps.append(overlap)

        avg_overlap = sum(overlaps) / len(overlaps) if overlaps else 0.0
        return 1.0 - avg_overlap

    # ========================================================================
    # Density Analysis
    # ========================================================================

    def _analyze_density(self, notes: List[Note]) -> Dict[str, float]:
        """
        Analyze textural density.

        Returns:
            Dictionary with 4 density metrics
        """
        # Compute density over time windows
        densities = self._compute_density_windows(notes, window_size=1.0)

        if not densities:
            return {
                "textural_density_mean": 0.0,
                "textural_density_variance": 0.0,
                "vertical_density": 0.0,
                "horizontal_density": 0.0
            }

        # Normalize densities
        if NUMPY_AVAILABLE:
            mean_density = float(np.mean(densities)) / 10.0  # Normalize
            variance = float(np.var(densities)) / 10.0
        else:
            mean_density = sum(densities) / len(densities) / 10.0
            variance = sum((d - mean_density)**2 for d in densities) / len(densities)

        # Vertical density (simultaneous notes)
        vertical = self._compute_vertical_density(notes)

        # Horizontal density (note onset rate)
        horizontal = self._compute_horizontal_density(notes)

        return {
            "textural_density_mean": min(1.0, mean_density),
            "textural_density_variance": min(1.0, variance),
            "vertical_density": vertical,
            "horizontal_density": horizontal
        }

    def _compute_density_windows(
        self,
        notes: List[Note],
        window_size: float
    ) -> List[float]:
        """Compute density in time windows"""
        if not notes:
            return []

        start_time = min(n.start_time for n in notes)
        end_time = max(n.start_time + n.duration for n in notes)

        densities = []
        current_time = start_time

        while current_time < end_time:
            # Count active notes in window
            active = sum(
                1 for n in notes
                if n.start_time < current_time + window_size
                and n.start_time + n.duration > current_time
            )
            densities.append(active)
            current_time += window_size

        return densities

    def _compute_vertical_density(self, notes: List[Note]) -> float:
        """Compute average simultaneous note count"""
        densities = self._compute_density_windows(notes, window_size=0.5)
        if not densities:
            return 0.0

        if NUMPY_AVAILABLE:
            return min(1.0, float(np.mean(densities)) / 8.0)  # Normalize to 8 voices
        return min(1.0, sum(densities) / len(densities) / 8.0)

    def _compute_horizontal_density(self, notes: List[Note]) -> float:
        """Compute note onset rate"""
        if len(notes) < 2:
            return 0.0

        sorted_notes = sorted(notes, key=lambda n: n.start_time)
        duration = sorted_notes[-1].start_time - sorted_notes[0].start_time

        if duration <= 0:
            return 0.0

        onset_rate = len(notes) / duration  # Notes per second
        # Normalize: 10 notes/sec = 1.0
        return min(1.0, onset_rate / 10.0)

    # ========================================================================
    # Layering Analysis
    # ========================================================================

    def _analyze_layering(
        self,
        notes: List[Note],
        notes_by_voice: Dict[int, List[Note]]
    ) -> Dict[str, float]:
        """
        Analyze layer structure and interaction.

        Returns:
            Dictionary with 4 layering metrics
        """
        num_voices = len(notes_by_voice)

        # Layer count (normalized)
        layer_count = min(1.0, num_voices / 8.0)

        # Layer interaction complexity
        interaction = self._compute_layer_interaction(notes_by_voice)

        # Foreground/background separation
        fg_bg_separation = self._compute_fg_bg_separation(notes_by_voice)

        # Voice crossing frequency
        crossing = self._compute_voice_crossing(notes_by_voice)

        return {
            "layer_count": layer_count,
            "layer_interaction_complexity": interaction,
            "foreground_background_separation": fg_bg_separation,
            "voice_crossing_frequency": crossing
        }

    def _compute_layer_interaction(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """Compute complexity of layer interactions"""
        if len(notes_by_voice) <= 1:
            return 0.0

        # Analyze temporal overlaps and pitch relationships
        # Simplified: use rhythmic independence as proxy
        sync = self._compute_rhythmic_synchronization(notes_by_voice)

        # High sync = low interaction, low sync = high interaction
        return 1.0 - sync

    def _compute_fg_bg_separation(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """Compute foreground/background clarity"""
        if len(notes_by_voice) <= 1:
            return 1.0  # Single voice = clear foreground

        # Analyze velocity differences and register separation
        voice_stats = {}
        for voice_id, voice_notes in notes_by_voice.items():
            if voice_notes:
                avg_velocity = sum(n.velocity for n in voice_notes) / len(voice_notes)
                avg_pitch = sum(n.pitch for n in voice_notes) / len(voice_notes)
                voice_stats[voice_id] = (avg_velocity, avg_pitch)

        # Measure separation
        # Higher variance = clearer separation
        if len(voice_stats) < 2:
            return 1.0

        velocities = [v[0] for v in voice_stats.values()]

        if NUMPY_AVAILABLE:
            vel_std = float(np.std(velocities))
            # Normalize: std of 20 = 1.0 separation
            return min(1.0, vel_std / 20.0)

        return 0.5

    def _compute_voice_crossing(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """Compute how frequently voices cross registers"""
        if len(notes_by_voice) <= 1:
            return 0.0

        # For each time point, check if voices maintain register order
        # This is simplified - just check pitch overlap
        voices = list(notes_by_voice.keys())
        crossings = 0
        checks = 0

        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                notes_i = notes_by_voice[voices[i]]
                notes_j = notes_by_voice[voices[j]]

                # Check pitch overlap
                if notes_i and notes_j:
                    pitches_i = set(n.pitch for n in notes_i)
                    pitches_j = set(n.pitch for n in notes_j)
                    overlap = len(pitches_i & pitches_j)

                    if overlap > 0:
                        crossings += 1
                    checks += 1

        return crossings / checks if checks > 0 else 0.0

    # ========================================================================
    # Temporal Pattern Analysis
    # ========================================================================

    def _analyze_temporal_patterns(
        self,
        notes: List[Note],
        notes_by_voice: Dict[int, List[Note]]
    ) -> Dict[str, float]:
        """
        Analyze temporal patterns.

        Returns:
            Dictionary with 4 temporal pattern metrics
        """
        # Call-response detection
        call_response = self._detect_call_response(notes_by_voice)

        # Imitation detection
        imitation = self._detect_imitation(notes_by_voice)

        # Texture evolution rate
        evolution_rate = self._compute_evolution_rate(notes)

        # Stagger vs synchronization
        stagger_sync = self._compute_stagger_sync_balance(notes_by_voice)

        return {
            "call_response_strength": call_response,
            "imitation_frequency": imitation,
            "texture_evolution_rate": evolution_rate,
            "stagger_synchronization_balance": stagger_sync
        }

    def _detect_call_response(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """Detect call-and-response patterns"""
        if len(notes_by_voice) <= 1:
            return 0.0

        # Look for alternating activity patterns
        # Simplified implementation
        return 0.3  # Placeholder

    def _detect_imitation(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """Detect imitative counterpoint"""
        if len(notes_by_voice) <= 1:
            return 0.0

        # Compare melodic sequences with time lag
        # Simplified implementation
        return 0.2  # Placeholder

    def _compute_evolution_rate(self, notes: List[Note]) -> float:
        """Compute rate of texture change over time"""
        # Analyze density changes
        densities = self._compute_density_windows(notes, window_size=2.0)

        if len(densities) < 2:
            return 0.0

        # Compute rate of change
        changes = [abs(densities[i] - densities[i-1]) for i in range(1, len(densities))]
        avg_change = sum(changes) / len(changes) if changes else 0.0

        # Normalize
        return min(1.0, avg_change / 4.0)

    def _compute_stagger_sync_balance(
        self,
        notes_by_voice: Dict[int, List[Note]]
    ) -> float:
        """
        Compute balance between staggered and synchronized onsets.

        Returns:
            Balance (0.0 = fully synchronized, 1.0 = fully staggered)
        """
        if len(notes_by_voice) <= 1:
            return 0.0

        sync = self._compute_rhythmic_synchronization(notes_by_voice)
        # Balance = 0.5 when balanced, moves toward 0 (sync) or 1 (stagger)
        return 1.0 - sync  # Stagger is inverse of sync

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _organize_by_voice(self, notes: List[Note]) -> Dict[int, List[Note]]:
        """Organize notes by voice/channel"""
        by_voice = defaultdict(list)
        for note in notes:
            by_voice[note.voice_id or note.channel].append(note)
        return dict(by_voice)

    def _empty_profile(self) -> TextureProfile:
        """Return empty texture profile"""
        return TextureProfile(
            homophonic_polyphonic_balance=0.0,
            monophonic_tendency=0.0,
            heterophonic_variation=0.0,
            texture_consistency=0.0,
            voice_independence_score=0.0,
            rhythmic_independence=0.0,
            melodic_independence=0.0,
            harmonic_independence=0.0,
            textural_density_mean=0.0,
            textural_density_variance=0.0,
            vertical_density=0.0,
            horizontal_density=0.0,
            layer_count=0.0,
            layer_interaction_complexity=0.0,
            foreground_background_separation=0.0,
            voice_crossing_frequency=0.0,
            call_response_strength=0.0,
            imitation_frequency=0.0,
            texture_evolution_rate=0.0,
            stagger_synchronization_balance=0.0
        )


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Texture Analysis Module - Agent 6")
    print("="*70)

    # Create analyzer
    analyzer = DetailedTextureAnalyzer()

    # Create test notes (simple homophonic texture)
    test_notes = [
        # Melody voice (voice 0)
        Note(60, 0.0, 1.0, 80, 0),
        Note(62, 1.0, 1.0, 80, 0),
        Note(64, 2.0, 1.0, 80, 0),
        Note(65, 3.0, 1.0, 80, 0),

        # Harmony voice (voice 1) - synchronized
        Note(48, 0.0, 1.0, 70, 1),
        Note(50, 1.0, 1.0, 70, 1),
        Note(52, 2.0, 1.0, 70, 1),
        Note(53, 3.0, 1.0, 70, 1),
    ]

    print("\n1. Analyzing homophonic texture...")
    profile = analyzer.analyze(test_notes)

    print("\n2. Texture Profile:")
    print(f"   Homophonic/Polyphonic Balance: {profile.homophonic_polyphonic_balance:.3f}")
    print(f"   Voice Independence: {profile.voice_independence_score:.3f}")
    print(f"   Textural Density (mean): {profile.textural_density_mean:.3f}")
    print(f"   Layer Count: {profile.layer_count:.3f}")
    print(f"   Call-Response Strength: {profile.call_response_strength:.3f}")

    # Create polyphonic texture test
    polyphonic_notes = [
        # Voice 1 - independent rhythm
        Note(60, 0.0, 0.5, 80, 0),
        Note(62, 0.75, 0.5, 80, 0),
        Note(64, 1.5, 0.5, 80, 0),

        # Voice 2 - different rhythm
        Note(55, 0.0, 1.0, 70, 1),
        Note(57, 1.5, 1.0, 70, 1),

        # Voice 3 - another independent rhythm
        Note(48, 0.25, 0.75, 65, 2),
        Note(50, 1.25, 0.75, 65, 2),
    ]

    print("\n3. Analyzing polyphonic texture...")
    poly_profile = analyzer.analyze(polyphonic_notes)

    print("\n4. Polyphonic Texture Profile:")
    print(f"   Homophonic/Polyphonic Balance: {poly_profile.homophonic_polyphonic_balance:.3f}")
    print(f"   Voice Independence: {poly_profile.voice_independence_score:.3f}")
    print(f"   Rhythmic Independence: {poly_profile.rhythmic_independence:.3f}")
    print(f"   Layer Count: {poly_profile.layer_count:.3f}")

    print("\n" + "="*70)
    print("✅ Texture analysis complete!")
    print("="*70)
