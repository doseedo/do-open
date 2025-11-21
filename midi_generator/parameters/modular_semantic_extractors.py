"""
Modular Semantic Parameter Extractors - Agent 3
================================================

Extracts 120 modular semantic parameters organized by musical dimension:
- Harmony (30 params)
- Rhythm (20 params)
- Form (15 params)
- Orchestration (25 params)
- Texture (20 params)
- Cross-dimensional (10 params)

These parameters are designed to be comprehensive, interpretable, and suitable
for training neural encoders.

Author: Agent 3 - Comprehensive Parameter Extraction Specialist
Date: November 21, 2025
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict
from dataclasses import dataclass
from scipy.stats import entropy
import warnings


@dataclass
class MIDIAnalysisData:
    """Shared MIDI analysis data for parameter extraction"""
    all_notes: List[Dict]
    melody_notes: List[Dict]
    chord_notes: List[List[Dict]]
    tracks_by_instrument: Dict[int, List[Dict]]
    all_pitches: List[int]
    melody_pitches: List[int]
    all_velocities: List[int]
    note_onsets: List[float]
    note_durations: List[float]
    tempo_bpm: float
    duration_seconds: float
    time_signature: str
    instrument_programs: List[int]


class HarmonyParameterExtractor:
    """Extract 30 comprehensive harmony parameters"""

    def extract(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Extract all harmony parameters"""
        params = {}

        # Basic chord properties (10 params)
        params['chord_density'] = self._compute_chord_density(analysis)
        params['chord_complexity'] = self._compute_chord_complexity(analysis)
        params['chord_voicing_spread'] = self._compute_voicing_spread(analysis)
        params['chord_root_variety'] = self._compute_root_variety(analysis)
        params['chord_inversion_ratio'] = self._compute_inversion_ratio(analysis)
        params['chord_extension_ratio'] = self._compute_extension_ratio(analysis)
        params['chord_sus_ratio'] = self._compute_suspended_ratio(analysis)
        params['chord_altered_ratio'] = self._compute_altered_ratio(analysis)
        params['chord_open_voicing_ratio'] = self._compute_open_voicing_ratio(analysis)
        params['chord_cluster_ratio'] = self._compute_cluster_ratio(analysis)

        # Harmonic movement (10 params)
        params['progression_strength'] = self._compute_progression_strength(analysis)
        params['progression_predictability'] = self._compute_progression_predictability(analysis)
        params['modulation_frequency'] = self._compute_modulation_frequency(analysis)
        params['chromatic_movement'] = self._compute_chromatic_movement(analysis)
        params['parallel_motion'] = self._compute_parallel_motion(analysis)
        params['contrary_motion'] = self._compute_contrary_motion(analysis)
        params['voice_crossing'] = self._compute_voice_crossing(analysis)
        params['bass_movement_stepwise'] = self._compute_bass_stepwise(analysis)
        params['upper_voice_independence'] = self._compute_voice_independence(analysis)
        params['harmonic_rhythm_rate'] = self._compute_harmonic_rhythm(analysis)

        # Harmonic color (10 params)
        params['dissonance_level'] = self._compute_dissonance_level(analysis)
        params['tension_curve_variance'] = self._compute_tension_variance(analysis)
        params['chromaticism'] = self._compute_chromaticism(analysis)
        params['modal_mixture'] = self._compute_modal_mixture(analysis)
        params['tritone_presence'] = self._compute_tritone_presence(analysis)
        params['dominant_preparation'] = self._compute_dominant_prep(analysis)
        params['pedal_point_ratio'] = self._compute_pedal_point(analysis)
        params['ostinato_ratio'] = self._compute_ostinato(analysis)
        params['harmonic_stability'] = self._compute_harmonic_stability(analysis)
        params['tonal_clarity'] = self._compute_tonal_clarity(analysis)

        return params

    def _compute_chord_density(self, analysis: MIDIAnalysisData) -> float:
        """Chords per measure"""
        if not analysis.chord_notes or analysis.duration_seconds == 0:
            return 0.0
        beats = analysis.duration_seconds * (analysis.tempo_bpm / 60.0)
        measures = beats / 4.0
        return float(len(analysis.chord_notes) / max(measures, 1))

    def _compute_chord_complexity(self, analysis: MIDIAnalysisData) -> float:
        """Average number of unique pitch classes per chord"""
        if not analysis.chord_notes:
            return 0.0
        complexities = []
        for chord in analysis.chord_notes:
            pitch_classes = set(n['pitch'] % 12 for n in chord)
            complexities.append(len(pitch_classes))
        return float(np.mean(complexities))

    def _compute_voicing_spread(self, analysis: MIDIAnalysisData) -> float:
        """Average pitch range of chords (normalized)"""
        if not analysis.chord_notes:
            return 0.5
        spreads = []
        for chord in analysis.chord_notes:
            pitches = [n['pitch'] for n in chord]
            if len(pitches) >= 2:
                spreads.append(max(pitches) - min(pitches))
        if not spreads:
            return 0.5
        return float(min(np.mean(spreads) / 36.0, 1.0))  # Normalize to 3 octaves

    def _compute_root_variety(self, analysis: MIDIAnalysisData) -> float:
        """Unique chord roots used / 12"""
        if not analysis.chord_notes:
            return 0.0
        roots = [min(n['pitch'] for n in chord) % 12 for chord in analysis.chord_notes]
        return float(len(set(roots)) / 12.0)

    def _compute_inversion_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of inverted chords"""
        # Simplified: detect if bass note is not the root
        # Would need full chord analysis for proper detection
        return 0.3  # Placeholder

    def _compute_extension_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of chords with extensions (7ths, 9ths, etc.)"""
        if not analysis.chord_notes:
            return 0.0
        extended_count = sum(1 for chord in analysis.chord_notes if len(chord) >= 4)
        return float(extended_count / len(analysis.chord_notes))

    def _compute_suspended_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of suspended chords"""
        # Simplified: detect sus4/sus2 intervals
        return 0.1  # Placeholder

    def _compute_altered_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of altered chords (b5, #5, etc.)"""
        # Simplified
        return 0.1  # Placeholder

    def _compute_open_voicing_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of open voicings (gaps > octave)"""
        if not analysis.chord_notes:
            return 0.5
        open_count = 0
        for chord in analysis.chord_notes:
            if len(chord) >= 2:
                pitches = sorted([n['pitch'] for n in chord])
                gaps = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
                if any(g > 12 for g in gaps):
                    open_count += 1
        return float(open_count / len(analysis.chord_notes))

    def _compute_cluster_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of cluster chords (adjacent semitones)"""
        if not analysis.chord_notes:
            return 0.0
        cluster_count = 0
        for chord in analysis.chord_notes:
            if len(chord) >= 2:
                pitches = sorted([n['pitch'] for n in chord])
                intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]
                if any(i <= 2 for i in intervals):
                    cluster_count += 1
        return float(cluster_count / len(analysis.chord_notes))

    def _compute_progression_strength(self, analysis: MIDIAnalysisData) -> float:
        """Strength of harmonic progressions (dominant motion)"""
        # Simplified: measure fourth/fifth bass movement
        if len(analysis.chord_notes) < 2:
            return 0.5
        roots = [min(n['pitch'] for n in chord) % 12 for chord in analysis.chord_notes]
        strong_movements = sum(1 for i in range(len(roots)-1)
                              if (roots[i+1] - roots[i]) % 12 in [5, 7])
        return float(strong_movements / (len(roots) - 1))

    def _compute_progression_predictability(self, analysis: MIDIAnalysisData) -> float:
        """Predictability via entropy of chord transitions"""
        if len(analysis.chord_notes) < 2:
            return 0.5
        roots = [min(n['pitch'] for n in chord) % 12 for chord in analysis.chord_notes]
        transitions = [(roots[i], roots[i+1]) for i in range(len(roots)-1)]
        if not transitions:
            return 0.5
        trans_counts = Counter(transitions)
        probs = np.array(list(trans_counts.values())) / len(transitions)
        trans_entropy = entropy(probs)
        max_entropy = np.log(len(trans_counts))
        return float(1.0 - (trans_entropy / max_entropy)) if max_entropy > 0 else 0.5

    def _compute_modulation_frequency(self, analysis: MIDIAnalysisData) -> float:
        """Frequency of key changes"""
        # Simplified: would need key detection over time
        return 0.2  # Placeholder

    def _compute_chromatic_movement(self, analysis: MIDIAnalysisData) -> float:
        """Chromatic voice leading ratio"""
        if len(analysis.chord_notes) < 2:
            return 0.0
        chromatic_count = 0
        for i in range(len(analysis.chord_notes) - 1):
            curr_pitches = set(n['pitch'] % 12 for n in analysis.chord_notes[i])
            next_pitches = set(n['pitch'] % 12 for n in analysis.chord_notes[i+1])
            # Check for semitone movement
            for p1 in curr_pitches:
                if (p1 + 1) % 12 in next_pitches or (p1 - 1) % 12 in next_pitches:
                    chromatic_count += 1
                    break
        return float(chromatic_count / (len(analysis.chord_notes) - 1))

    def _compute_parallel_motion(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of parallel voice motion"""
        return 0.3  # Placeholder - needs voice tracking

    def _compute_contrary_motion(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of contrary voice motion"""
        return 0.4  # Placeholder - needs voice tracking

    def _compute_voice_crossing(self, analysis: MIDIAnalysisData) -> float:
        """Frequency of voice crossing"""
        return 0.1  # Placeholder

    def _compute_bass_stepwise(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of stepwise bass movement"""
        if len(analysis.chord_notes) < 2:
            return 0.5
        roots = [min(n['pitch'] for n in chord) for chord in analysis.chord_notes]
        stepwise = sum(1 for i in range(len(roots)-1)
                      if abs(roots[i+1] - roots[i]) <= 2)
        return float(stepwise / (len(roots) - 1))

    def _compute_voice_independence(self, analysis: MIDIAnalysisData) -> float:
        """Independence of upper voices"""
        return 0.5  # Placeholder - needs voice tracking

    def _compute_harmonic_rhythm(self, analysis: MIDIAnalysisData) -> float:
        """Harmonic rhythm rate (chord changes per beat)"""
        if not analysis.chord_notes or analysis.duration_seconds == 0:
            return 0.0
        beats = analysis.duration_seconds * (analysis.tempo_bpm / 60.0)
        return float(len(analysis.chord_notes) / max(beats, 1))

    def _compute_dissonance_level(self, analysis: MIDIAnalysisData) -> float:
        """Average dissonance level"""
        if not analysis.chord_notes:
            return 0.0
        dissonances = []
        for chord in analysis.chord_notes:
            if len(chord) < 2:
                continue
            pitches = sorted([n['pitch'] for n in chord])
            intervals = [(pitches[j] - pitches[i]) % 12
                        for i in range(len(pitches))
                        for j in range(i+1, len(pitches))]
            # Dissonance weights
            diss_map = {1: 1.0, 2: 0.8, 6: 0.6, 10: 0.4, 11: 0.9, 0: 0.0,
                       3: 0.2, 4: 0.3, 5: 0.1, 7: 0.2, 8: 0.3, 9: 0.4}
            chord_diss = np.mean([diss_map.get(i, 0.5) for i in intervals])
            dissonances.append(chord_diss)
        return float(np.mean(dissonances)) if dissonances else 0.0

    def _compute_tension_variance(self, analysis: MIDIAnalysisData) -> float:
        """Variance in harmonic tension over time"""
        # Simplified
        return 0.3  # Placeholder

    def _compute_chromaticism(self, analysis: MIDIAnalysisData) -> float:
        """Chromaticism level (unique pitch classes / 12)"""
        if not analysis.all_pitches:
            return 0.0
        pitch_classes = set(p % 12 for p in analysis.all_pitches)
        return float(len(pitch_classes) / 12.0)

    def _compute_modal_mixture(self, analysis: MIDIAnalysisData) -> float:
        """Modal mixture / borrowed chords"""
        return 0.2  # Placeholder - needs key analysis

    def _compute_tritone_presence(self, analysis: MIDIAnalysisData) -> float:
        """Tritone presence ratio"""
        if not analysis.chord_notes:
            return 0.0
        tritone_count = 0
        for chord in analysis.chord_notes:
            pitches = [n['pitch'] % 12 for n in chord]
            for i, p1 in enumerate(pitches):
                for p2 in pitches[i+1:]:
                    if abs(p2 - p1) % 12 == 6:
                        tritone_count += 1
                        break
        return float(min(tritone_count / len(analysis.chord_notes), 1.0))

    def _compute_dominant_prep(self, analysis: MIDIAnalysisData) -> float:
        """Dominant preparation ratio"""
        return 0.3  # Placeholder

    def _compute_pedal_point(self, analysis: MIDIAnalysisData) -> float:
        """Pedal point ratio"""
        return 0.2  # Placeholder

    def _compute_ostinato(self, analysis: MIDIAnalysisData) -> float:
        """Harmonic ostinato ratio"""
        if len(analysis.chord_notes) < 4:
            return 0.0
        # Detect repeating chord patterns
        patterns = []
        for i in range(len(analysis.chord_notes) - 1):
            root1 = min(n['pitch'] for n in analysis.chord_notes[i]) % 12
            root2 = min(n['pitch'] for n in analysis.chord_notes[i+1]) % 12
            patterns.append((root1, root2))
        if not patterns:
            return 0.0
        pattern_counts = Counter(patterns)
        most_common_count = pattern_counts.most_common(1)[0][1]
        return float(min(most_common_count / len(patterns), 1.0))

    def _compute_harmonic_stability(self, analysis: MIDIAnalysisData) -> float:
        """Harmonic stability (consonance)"""
        return 1.0 - self._compute_dissonance_level(analysis)

    def _compute_tonal_clarity(self, analysis: MIDIAnalysisData) -> float:
        """Tonal clarity / ambiguity"""
        # Based on pitch class distribution clarity
        if not analysis.all_pitches:
            return 0.5
        pitch_classes = [p % 12 for p in analysis.all_pitches]
        pc_counts = [pitch_classes.count(i) for i in range(12)]
        total = sum(pc_counts)
        if total == 0:
            return 0.5
        pc_dist = [c / total for c in pc_counts]
        pc_entropy = entropy(pc_dist)
        max_entropy = np.log(12)
        # High entropy = low clarity
        return float(1.0 - (pc_entropy / max_entropy))


class RhythmParameterExtractor:
    """Extract 20 comprehensive rhythm parameters"""

    def extract(self, analysis: MIDIAnalysisData) -> Dict[str, float]:
        """Extract all rhythm parameters"""
        params = {}

        # Basic rhythm (7 params)
        params['note_density'] = self._compute_note_density(analysis)
        params['subdivision_level'] = self._compute_subdivision_level(analysis)
        params['syncopation'] = self._compute_syncopation(analysis)
        params['groove_consistency'] = self._compute_groove_consistency(analysis)
        params['swing_amount'] = self._compute_swing_amount(analysis)
        params['polyrhythm_level'] = self._compute_polyrhythm(analysis)
        params['metric_complexity'] = self._compute_metric_complexity(analysis)

        # Rhythmic patterns (7 params)
        params['pattern_repetition'] = self._compute_pattern_repetition(analysis)
        params['rhythmic_diversity'] = self._compute_rhythmic_diversity(analysis)
        params['rhythmic_density_variance'] = self._compute_density_variance(analysis)
        params['long_note_ratio'] = self._compute_long_note_ratio(analysis)
        params['staccato_ratio'] = self._compute_staccato_ratio(analysis)
        params['rest_ratio'] = self._compute_rest_ratio(analysis)
        params['anacrusis_presence'] = self._compute_anacrusis(analysis)

        # Advanced rhythm (6 params)
        params['hemiola_presence'] = self._compute_hemiola(analysis)
        params['cross_rhythm'] = self._compute_cross_rhythm(analysis)
        params['rhythmic_acceleration'] = self._compute_acceleration(analysis)
        params['rubato_amount'] = self._compute_rubato(analysis)
        params['micro_timing_variance'] = self._compute_micro_timing(analysis)
        params['quantization_level'] = self._compute_quantization(analysis)

        return params

    def _compute_note_density(self, analysis: MIDIAnalysisData) -> float:
        """Notes per second"""
        if analysis.duration_seconds == 0:
            return 0.0
        return float(len(analysis.all_notes) / analysis.duration_seconds)

    def _compute_subdivision_level(self, analysis: MIDIAnalysisData) -> float:
        """Subdivision level (0=whole, 1=half, 2=quarter, 3=eighth, 4=sixteenth)"""
        if not analysis.note_durations:
            return 2.0
        min_duration = min(analysis.note_durations)
        beat_duration = 60.0 / analysis.tempo_bpm
        if min_duration >= beat_duration:
            return 2.0  # Quarter note
        elif min_duration >= beat_duration / 2:
            return 3.0  # Eighth note
        elif min_duration >= beat_duration / 4:
            return 4.0  # Sixteenth note
        else:
            return 5.0  # Thirty-second or smaller

    def _compute_syncopation(self, analysis: MIDIAnalysisData) -> float:
        """Syncopation level (off-beat emphasis)"""
        if not analysis.note_onsets:
            return 0.0
        beat_duration = 60.0 / analysis.tempo_bpm
        off_beat = 0
        for onset in analysis.note_onsets:
            beat_pos = (onset % beat_duration) / beat_duration
            if 0.2 < beat_pos < 0.8:  # Off-beat
                off_beat += 1
        return float(off_beat / len(analysis.note_onsets))

    def _compute_groove_consistency(self, analysis: MIDIAnalysisData) -> float:
        """Rhythmic groove consistency"""
        if len(analysis.note_onsets) < 2:
            return 1.0
        iois = [analysis.note_onsets[i+1] - analysis.note_onsets[i]
                for i in range(len(analysis.note_onsets)-1)]
        if not iois:
            return 1.0
        std = np.std(iois)
        mean = np.mean(iois)
        if mean == 0:
            return 1.0
        cv = std / mean
        return float(np.clip(1.0 - cv, 0.0, 1.0))

    def _compute_swing_amount(self, analysis: MIDIAnalysisData) -> float:
        """Swing ratio (0.5=straight, 0.67=triplet swing)"""
        # Simplified swing detection
        return 0.5  # Placeholder

    def _compute_polyrhythm(self, analysis: MIDIAnalysisData) -> float:
        """Polyrhythm complexity"""
        if len(analysis.tracks_by_instrument) < 2:
            return 0.0
        # Compare rhythm entropy across tracks
        entropies = []
        for prog, notes in analysis.tracks_by_instrument.items():
            if len(notes) > 1:
                durations = [n['duration'] for n in notes]
                dur_counts = Counter([round(d, 2) for d in durations])
                probs = np.array(list(dur_counts.values())) / len(durations)
                entropies.append(entropy(probs))
        if len(entropies) < 2:
            return 0.0
        return float(min(np.var(entropies), 1.0))

    def _compute_metric_complexity(self, analysis: MIDIAnalysisData) -> float:
        """Metric complexity (time signature complexity)"""
        time_sig = analysis.time_signature
        if time_sig in ['4/4', '2/4', '3/4']:
            return 0.0  # Simple
        elif time_sig in ['6/8', '12/8', '9/8']:
            return 0.3  # Compound
        elif time_sig in ['5/4', '7/8', '7/4']:
            return 0.7  # Irregular
        else:
            return 0.5  # Unknown

    def _compute_pattern_repetition(self, analysis: MIDIAnalysisData) -> float:
        """Rhythmic pattern repetition"""
        if len(analysis.note_durations) < 8:
            return 0.0
        # Find repeating duration patterns
        pattern_len = 4
        patterns = [tuple(analysis.note_durations[i:i+pattern_len])
                   for i in range(len(analysis.note_durations) - pattern_len + 1)]
        if not patterns:
            return 0.0
        pattern_counts = Counter(patterns)
        most_common = pattern_counts.most_common(1)[0][1]
        return float(min(most_common / len(patterns), 1.0))

    def _compute_rhythmic_diversity(self, analysis: MIDIAnalysisData) -> float:
        """Diversity of rhythm durations (entropy)"""
        if not analysis.note_durations:
            return 0.0
        dur_buckets = [round(d * 4) / 4 for d in analysis.note_durations]
        dur_counts = Counter(dur_buckets)
        probs = np.array(list(dur_counts.values())) / len(analysis.note_durations)
        dur_entropy = entropy(probs)
        max_entropy = np.log(min(len(dur_counts), 8))
        return float(dur_entropy / max_entropy) if max_entropy > 0 else 0.0

    def _compute_density_variance(self, analysis: MIDIAnalysisData) -> float:
        """Variance in rhythmic density over time"""
        # Simplified
        return 0.3  # Placeholder

    def _compute_long_note_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of long notes (> 1 beat)"""
        if not analysis.note_durations:
            return 0.0
        beat_duration = 60.0 / analysis.tempo_bpm
        long_notes = sum(1 for d in analysis.note_durations if d > beat_duration)
        return float(long_notes / len(analysis.note_durations))

    def _compute_staccato_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of staccato articulation"""
        # Detect short notes with gaps
        if not analysis.all_notes or len(analysis.all_notes) < 2:
            return 0.0
        sorted_notes = sorted(analysis.all_notes, key=lambda n: n['onset'])
        staccato_count = 0
        for i in range(len(sorted_notes) - 1):
            duration = sorted_notes[i]['duration']
            ioi = sorted_notes[i+1]['onset'] - sorted_notes[i]['onset']
            if ioi > 0 and duration / ioi < 0.5:  # Less than 50% legato
                staccato_count += 1
        return float(staccato_count / (len(sorted_notes) - 1))

    def _compute_rest_ratio(self, analysis: MIDIAnalysisData) -> float:
        """Ratio of silence to sound"""
        if not analysis.all_notes or analysis.duration_seconds == 0:
            return 0.0
        total_sound = sum(n['duration'] for n in analysis.all_notes)
        return float(1.0 - min(total_sound / analysis.duration_seconds, 1.0))

    def _compute_anacrusis(self, analysis: MIDIAnalysisData) -> float:
        """Anacrusis presence (pickup notes)"""
        # Check if first note starts before first downbeat
        if not analysis.note_onsets:
            return 0.0
        first_onset = analysis.note_onsets[0]
        beat_duration = 60.0 / analysis.tempo_bpm
        if first_onset < beat_duration * 0.9:
            return 1.0
        return 0.0

    def _compute_hemiola(self, analysis: MIDIAnalysisData) -> float:
        """Hemiola pattern presence"""
        return 0.1  # Placeholder

    def _compute_cross_rhythm(self, analysis: MIDIAnalysisData) -> float:
        """Cross-rhythm complexity"""
        return self._compute_polyrhythm(analysis)

    def _compute_acceleration(self, analysis: MIDIAnalysisData) -> float:
        """Rhythmic acceleration (tempo changes)"""
        return 0.0  # Placeholder - would need tempo tracking

    def _compute_rubato(self, analysis: MIDIAnalysisData) -> float:
        """Rubato amount (expressive timing)"""
        return 0.0  # Placeholder

    def _compute_micro_timing(self, analysis: MIDIAnalysisData) -> float:
        """Micro-timing variance (humanization)"""
        if len(analysis.note_onsets) < 2:
            return 0.0
        beat_duration = 60.0 / analysis.tempo_bpm
        # Measure deviation from grid
        deviations = []
        for onset in analysis.note_onsets:
            grid_pos = round(onset / beat_duration) * beat_duration
            deviation = abs(onset - grid_pos)
            deviations.append(deviation)
        return float(np.mean(deviations) / beat_duration) if deviations else 0.0

    def _compute_quantization(self, analysis: MIDIAnalysisData) -> float:
        """Quantization level (1=fully quantized, 0=no quantization)"""
        return 1.0 - self._compute_micro_timing(analysis)


# Additional extractors will be implemented in separate files for clarity
# Form, Orchestration, Texture, and Cross-dimensional extractors
