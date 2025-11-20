"""
Hierarchical Parameter Extractor - Agent 01 Phase 2
====================================================

Extracts the 50 hierarchical parameters from MIDI files.

This module implements the complete extraction pipeline for the new
hierarchical parameter system:
- Level 1: Global Context (8 parameters)
- Level 2: Universal Dimensions (20 parameters)
- Level 3: Genre-Specific Details (22 parameters)

Author: Agent 01 - Parameter Consolidation Architect
Date: November 20, 2025
Version: 2.0.0
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import mido
from collections import Counter, defaultdict
from scipy.stats import entropy
import warnings


@dataclass
class MIDIAnalysis:
    """Container for raw MIDI analysis data"""
    midi_file: mido.MidiFile
    total_notes: int = 0
    duration_seconds: float = 0.0
    tempo_bpm: float = 120.0
    time_signature: str = "4/4"

    # Note data
    all_notes: List[Dict] = field(default_factory=list)
    melody_notes: List[Dict] = field(default_factory=list)
    chord_notes: List[List[Dict]] = field(default_factory=list)

    # Track data
    tracks_by_instrument: Dict[int, List[Dict]] = field(default_factory=dict)
    instrument_programs: List[int] = field(default_factory=list)

    # Timing data
    note_onsets: List[float] = field(default_factory=list)
    note_durations: List[float] = field(default_factory=list)

    # Pitch data
    all_pitches: List[int] = field(default_factory=list)
    melody_pitches: List[int] = field(default_factory=list)

    # Velocity data
    all_velocities: List[int] = field(default_factory=list)


class HierarchicalParameterExtractor:
    """
    Extracts 50 hierarchical parameters from MIDI files.

    Usage:
        extractor = HierarchicalParameterExtractor()
        params = extractor.extract_from_midi("path/to/file.mid")
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._load_hierarchical_schema()

    def _load_hierarchical_schema(self):
        """Load the hierarchical parameter schema"""
        schema_path = Path(__file__).parent / "hierarchical_parameters.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        else:
            warnings.warn("hierarchical_parameters.json not found, using defaults")
            self.schema = {}

    def extract_from_midi(self, midi_path: str) -> Dict[str, Any]:
        """
        Extract all 50 hierarchical parameters from a MIDI file.

        Args:
            midi_path: Path to MIDI file

        Returns:
            Dictionary with Level 1, 2, and 3 parameters
        """
        if self.verbose:
            print(f"Extracting parameters from: {midi_path}")

        # Stage 1: Analyze MIDI file
        analysis = self._analyze_midi(midi_path)

        # Stage 2: Extract Level 2 (Universal Dimensions)
        level2 = self._extract_level2(analysis)

        # Stage 3: Extract Level 1 (Global Context) - depends on Level 2
        level1 = self._extract_level1(analysis, level2)

        # Stage 4: Extract Level 3 (Genre-Specific) - depends on Level 1
        level3 = self._extract_level3(analysis, level1, level2)

        return {
            "level1_global": level1,
            "level2_universal": level2,
            "level3_genre_specific": level3,
            "metadata": {
                "file": str(midi_path),
                "extraction_version": "2.0.0",
                "total_notes": analysis.total_notes,
                "duration_seconds": analysis.duration_seconds
            }
        }

    # ========================================================================
    # MIDI Analysis
    # ========================================================================

    def _analyze_midi(self, midi_path: str) -> MIDIAnalysis:
        """Analyze MIDI file and extract raw data"""
        midi_file = mido.MidiFile(midi_path)
        analysis = MIDIAnalysis(midi_file=midi_file)

        # Extract tempo and time signature from meta messages
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    analysis.tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    analysis.time_signature = f"{msg.numerator}/{msg.denominator}"

        # Calculate duration
        analysis.duration_seconds = midi_file.length

        # Extract all notes with timing and velocity
        current_time = 0.0
        active_notes = {}  # track: {pitch: (onset_time, velocity)}

        for track_idx, track in enumerate(midi_file.tracks):
            current_time = 0.0
            instrument_program = 0

            for msg in track:
                current_time += mido.tick2second(msg.time, midi_file.ticks_per_beat,
                                                 mido.bpm2tempo(analysis.tempo_bpm))

                if msg.type == 'program_change':
                    instrument_program = msg.program
                    if instrument_program not in analysis.instrument_programs:
                        analysis.instrument_programs.append(instrument_program)

                elif msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_idx, msg.channel, msg.note)
                    active_notes[key] = (current_time, msg.velocity)

                elif msg.type in ['note_off', 'note_on']:  # note_on with velocity 0 = note_off
                    if msg.type == 'note_on' and msg.velocity > 0:
                        continue

                    key = (track_idx, msg.channel, msg.note)
                    if key in active_notes:
                        onset_time, velocity = active_notes[key]
                        duration = current_time - onset_time

                        note_data = {
                            'pitch': msg.note,
                            'velocity': velocity,
                            'onset': onset_time,
                            'duration': duration,
                            'track': track_idx,
                            'channel': msg.channel,
                            'program': instrument_program
                        }

                        analysis.all_notes.append(note_data)
                        analysis.all_pitches.append(msg.note)
                        analysis.all_velocities.append(velocity)
                        analysis.note_onsets.append(onset_time)
                        analysis.note_durations.append(duration)

                        # Organize by instrument
                        if instrument_program not in analysis.tracks_by_instrument:
                            analysis.tracks_by_instrument[instrument_program] = []
                        analysis.tracks_by_instrument[instrument_program].append(note_data)

                        del active_notes[key]

        analysis.total_notes = len(analysis.all_notes)

        # Identify melody track (highest average pitch, typically)
        if analysis.tracks_by_instrument:
            melody_program = max(analysis.tracks_by_instrument.keys(),
                               key=lambda p: np.mean([n['pitch'] for n in analysis.tracks_by_instrument[p]]))
            analysis.melody_notes = analysis.tracks_by_instrument[melody_program]
            analysis.melody_pitches = [n['pitch'] for n in analysis.melody_notes]

        # Detect simultaneous notes (chords)
        analysis.chord_notes = self._detect_chords(analysis.all_notes)

        return analysis

    def _detect_chords(self, notes: List[Dict], time_window: float = 0.05) -> List[List[Dict]]:
        """Detect simultaneous notes (chords) within time window"""
        if not notes:
            return []

        sorted_notes = sorted(notes, key=lambda n: n['onset'])
        chords = []
        current_chord = [sorted_notes[0]]

        for note in sorted_notes[1:]:
            if abs(note['onset'] - current_chord[0]['onset']) < time_window:
                current_chord.append(note)
            else:
                if len(current_chord) >= 2:  # At least 2 notes for a chord
                    chords.append(current_chord)
                current_chord = [note]

        if len(current_chord) >= 2:
            chords.append(current_chord)

        return chords

    # ========================================================================
    # LEVEL 1: Global Context (8 parameters)
    # ========================================================================

    def _extract_level1(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract Level 1 global context parameters"""
        level1 = {}

        # tempo.bpm - from MIDI meta events
        level1['tempo.bpm'] = float(analysis.tempo_bpm)

        # time_signature - from MIDI meta events
        level1['time_signature'] = analysis.time_signature

        # key.tonic and key.mode - key detection
        tonic, mode = self._detect_key(analysis.all_pitches)
        level1['key.tonic'] = tonic
        level1['key.mode'] = mode

        # genre.primary - genre classification
        level1['genre.primary'] = self._classify_genre(analysis, level2)

        # structure.form - form analysis
        level1['structure.form'] = self._detect_form(analysis)

        # energy.level - aggregate of dynamics, tempo, texture
        level1['energy.level'] = self._compute_energy_level(
            analysis.tempo_bpm,
            level2['dynamics']['overall_level'],
            level2['texture']['density']
        )

        # complexity.overall - aggregate of harmony and melody complexity
        level1['complexity.overall'] = self._compute_overall_complexity(level2)

        return level1

    def _detect_key(self, pitches: List[int]) -> Tuple[str, str]:
        """Detect key using Krumhansl-Schmuckler algorithm"""
        if not pitches:
            return "C", "major"

        # Major and minor profiles (Krumhansl-Schmuckler)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        pitch_classes = [p % 12 for p in pitches]
        pitch_class_counts = [pitch_classes.count(i) for i in range(12)]

        # Normalize
        total = sum(pitch_class_counts)
        if total == 0:
            return "C", "major"
        pitch_class_dist = [c / total for c in pitch_class_counts]

        # Correlate with each key
        key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        best_corr = -1
        best_key = 'C'
        best_mode = 'major'

        for shift in range(12):
            # Major
            shifted_profile = major_profile[shift:] + major_profile[:shift]
            corr = np.corrcoef(pitch_class_dist, shifted_profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = key_names[shift]
                best_mode = 'major'

            # Minor
            shifted_profile = minor_profile[shift:] + minor_profile[:shift]
            corr = np.corrcoef(pitch_class_dist, shifted_profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = key_names[shift]
                best_mode = 'minor'

        return best_key, best_mode

    def _classify_genre(self, analysis: MIDIAnalysis, level2: Dict) -> str:
        """Classify genre based on musical features"""
        # Simple heuristic-based genre classification
        # In production, this would use a trained ML classifier

        tempo = analysis.tempo_bpm
        swing = level2['rhythm']['swing_amount']
        syncopation = level2['rhythm']['syncopation']
        chord_density = level2['harmony']['chord_density']

        # Jazz indicators
        if swing > 0.6 and chord_density > 4.0:
            return "jazz"

        # Classical indicators
        if len(analysis.instrument_programs) > 5 and tempo < 140:
            # Check for orchestral instruments
            orchestral_programs = [40, 41, 42, 56, 57, 58, 60, 68, 69, 70, 71]
            if any(p in orchestral_programs for p in analysis.instrument_programs):
                return "classical"

        # Electronic indicators (high quantization would be detected here)
        if len(analysis.chord_notes) > 10 and tempo > 120:
            return "electronic"

        # Rock indicators (power chords would be detected)
        if 29 <= analysis.instrument_programs[0] <= 31 if analysis.instrument_programs else False:
            return "rock"

        # Default to pop
        return "pop"

    def _detect_form(self, analysis: MIDIAnalysis) -> str:
        """Detect musical form structure"""
        # Simplified form detection - would be more sophisticated in production
        duration = analysis.duration_seconds

        if duration < 60:
            return "AABA"
        elif duration < 180:
            return "verse_chorus"
        else:
            return "through_composed"

    def _compute_energy_level(self, tempo: float, dynamics: float, density: float) -> float:
        """Compute overall energy level"""
        tempo_norm = min(tempo / 200.0, 1.0)
        energy = 0.3 * dynamics + 0.3 * tempo_norm + 0.4 * min(density / 10.0, 1.0)
        return float(np.clip(energy, 0.0, 1.0))

    def _compute_overall_complexity(self, level2: Dict) -> float:
        """Compute overall musical complexity"""
        harmony_complexity = level2['harmony']['complexity']
        melody_complexity = level2['melody']['rhythmic_complexity']
        rhythm_syncopation = level2['rhythm']['syncopation']

        complexity = (0.5 * harmony_complexity +
                     0.3 * melody_complexity +
                     0.2 * rhythm_syncopation)
        return float(np.clip(complexity, 0.0, 1.0))

    # ========================================================================
    # LEVEL 2: Universal Dimensions (20 parameters)
    # ========================================================================

    def _extract_level2(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract Level 2 universal dimension parameters"""
        level2 = {
            'harmony': self._extract_harmony(analysis),
            'melody': self._extract_melody(analysis),
            'rhythm': self._extract_rhythm(analysis),
            'dynamics': self._extract_dynamics(analysis),
            'texture': self._extract_texture(analysis)
        }
        return level2

    def _extract_harmony(self, analysis: MIDIAnalysis) -> Dict[str, float]:
        """Extract 6 harmony parameters"""
        harmony = {}

        # chord_density: chords per measure × notes per chord
        if analysis.chord_notes and analysis.duration_seconds > 0:
            beats = analysis.duration_seconds * (analysis.tempo_bpm / 60.0)
            measures = beats / 4.0  # Assuming 4/4
            chords_per_measure = len(analysis.chord_notes) / max(measures, 1)
            avg_notes_per_chord = np.mean([len(chord) for chord in analysis.chord_notes])
            harmony['chord_density'] = float(chords_per_measure * avg_notes_per_chord)
        else:
            harmony['chord_density'] = 0.0

        # complexity: based on chord extensions (heuristic)
        # In full implementation, would analyze actual chord qualities
        harmony['complexity'] = min(harmony['chord_density'] / 10.0, 1.0)

        # chromaticism: chromatic notes ratio
        if analysis.all_pitches:
            pitch_classes = [p % 12 for p in analysis.all_pitches]
            # Simple heuristic: variety of pitch classes used
            unique_pcs = len(set(pitch_classes))
            harmony['chromaticism'] = float(unique_pcs / 12.0)
        else:
            harmony['chromaticism'] = 0.0

        # tension: based on dissonance (simplified)
        harmony['tension'] = self._compute_harmonic_tension(analysis.chord_notes)

        # voicing_spread: average chord range
        if analysis.chord_notes:
            spreads = [(max(n['pitch'] for n in chord) - min(n['pitch'] for n in chord))
                      for chord in analysis.chord_notes]
            avg_spread = np.mean(spreads)
            harmony['voicing_spread'] = float(min(avg_spread / 36.0, 1.0))
        else:
            harmony['voicing_spread'] = 0.5

        # progression_predictability: entropy of chord transitions
        harmony['progression_predictability'] = self._compute_progression_predictability(analysis.chord_notes)

        return harmony

    def _compute_harmonic_tension(self, chords: List[List[Dict]]) -> float:
        """Compute average harmonic tension based on intervals"""
        if not chords:
            return 0.0

        tensions = []
        for chord in chords:
            if len(chord) < 2:
                continue
            pitches = sorted([n['pitch'] for n in chord])
            intervals = [pitches[i+1] - pitches[i] for i in range(len(pitches)-1)]

            # Dissonance weights (semitones)
            dissonance = {1: 1.0, 2: 0.8, 6: 0.6, 10: 0.4, 11: 0.9}
            chord_tension = sum(dissonance.get(i % 12, 0.2) for i in intervals) / len(intervals)
            tensions.append(chord_tension)

        return float(np.mean(tensions)) if tensions else 0.0

    def _compute_progression_predictability(self, chords: List[List[Dict]]) -> float:
        """Compute progression predictability via entropy"""
        if len(chords) < 2:
            return 0.5

        # Simplified: measure root movement patterns
        roots = [min(n['pitch'] for n in chord) % 12 for chord in chords]
        transitions = [roots[i+1] - roots[i] for i in range(len(roots)-1)]

        if not transitions:
            return 0.5

        # Calculate entropy of transitions
        transition_counts = Counter(transitions)
        probs = np.array(list(transition_counts.values())) / len(transitions)
        transition_entropy = entropy(probs)

        # Normalize: lower entropy = more predictable
        max_entropy = np.log(12)  # Maximum for 12 possible transitions
        predictability = 1.0 - (transition_entropy / max_entropy)

        return float(np.clip(predictability, 0.0, 1.0))

    def _extract_melody(self, analysis: MIDIAnalysis) -> Dict[str, float]:
        """Extract 5 melody parameters"""
        melody = {}

        if not analysis.melody_notes:
            return {
                'note_density': 0.0,
                'range_semitones': 0,
                'contour_smoothness': 0.0,
                'rhythmic_complexity': 0.0,
                'repetition': 0.0
            }

        # note_density: notes per measure
        beats = analysis.duration_seconds * (analysis.tempo_bpm / 60.0)
        measures = beats / 4.0
        melody['note_density'] = float(len(analysis.melody_notes) / max(measures, 1))

        # range_semitones: pitch range
        melody['range_semitones'] = int(max(analysis.melody_pitches) - min(analysis.melody_pitches))

        # contour_smoothness: stepwise motion ratio × leap size factor
        intervals = [abs(analysis.melody_pitches[i+1] - analysis.melody_pitches[i])
                    for i in range(len(analysis.melody_pitches)-1)]
        if intervals:
            stepwise_ratio = sum(1 for i in intervals if i <= 2) / len(intervals)
            avg_leap = np.mean(intervals)
            melody['contour_smoothness'] = float(stepwise_ratio * (1 - min(avg_leap / 12.0, 1.0)))
        else:
            melody['contour_smoothness'] = 0.0

        # rhythmic_complexity: entropy of note durations
        durations = [n['duration'] for n in analysis.melody_notes]
        melody['rhythmic_complexity'] = self._compute_rhythm_entropy(durations)

        # repetition: repeated motif ratio (simplified)
        melody['repetition'] = self._detect_melodic_repetition(analysis.melody_pitches)

        return melody

    def _compute_rhythm_entropy(self, durations: List[float]) -> float:
        """Compute rhythmic complexity via entropy"""
        if not durations:
            return 0.0

        # Quantize durations to discrete buckets
        duration_buckets = [round(d * 4) / 4 for d in durations]  # Quarter note resolution
        duration_counts = Counter(duration_buckets)

        if len(duration_counts) == 1:
            return 0.0  # No variety

        probs = np.array(list(duration_counts.values())) / len(durations)
        rhythm_entropy = entropy(probs)

        # Normalize
        max_entropy = np.log(min(len(duration_counts), 8))  # Cap at 8 duration types
        return float(min(rhythm_entropy / max_entropy, 1.0)) if max_entropy > 0 else 0.0

    def _detect_melodic_repetition(self, pitches: List[int], motif_length: int = 4) -> float:
        """Detect repeated melodic motifs"""
        if len(pitches) < motif_length * 2:
            return 0.0

        motifs = [tuple(pitches[i:i+motif_length])
                 for i in range(len(pitches) - motif_length + 1)]
        motif_counts = Counter(motifs)

        repeated_motifs = sum(count - 1 for count in motif_counts.values() if count > 1)
        total_motifs = len(motifs)

        return float(repeated_motifs / total_motifs) if total_motifs > 0 else 0.0

    def _extract_rhythm(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract 5 rhythm parameters"""
        rhythm = {}

        # subdivision: detect smallest duration
        if analysis.note_durations:
            min_duration = min(analysis.note_durations)
            beat_duration = 60.0 / analysis.tempo_bpm

            if min_duration >= beat_duration * 0.9:
                rhythm['subdivision'] = 'quarter'
            elif min_duration >= beat_duration * 0.4:
                rhythm['subdivision'] = 'eighth'
            elif min_duration >= beat_duration * 0.3:
                rhythm['subdivision'] = 'triplet'
            else:
                rhythm['subdivision'] = 'sixteenth'
        else:
            rhythm['subdivision'] = 'quarter'

        # syncopation: off-beat ratio
        rhythm['syncopation'] = self._compute_syncopation(analysis)

        # groove_consistency: timing stability
        rhythm['groove_consistency'] = self._compute_groove_consistency(analysis)

        # polyrhythm: detect conflicting rhythms
        rhythm['polyrhythm'] = self._detect_polyrhythm(analysis)

        # swing_amount: measure swing ratio
        rhythm['swing_amount'] = self._compute_swing_amount(analysis)

        return rhythm

    def _compute_syncopation(self, analysis: MIDIAnalysis) -> float:
        """Compute syncopation level"""
        if not analysis.note_onsets:
            return 0.0

        beat_duration = 60.0 / analysis.tempo_bpm

        # Count notes on vs off beats
        on_beat_count = 0
        off_beat_count = 0

        for onset in analysis.note_onsets:
            beat_position = (onset % beat_duration) / beat_duration
            if beat_position < 0.1 or beat_position > 0.9:  # Close to beat
                on_beat_count += 1
            else:
                off_beat_count += 1

        total = on_beat_count + off_beat_count
        return float(off_beat_count / total) if total > 0 else 0.0

    def _compute_groove_consistency(self, analysis: MIDIAnalysis) -> float:
        """Compute rhythmic stability"""
        if len(analysis.note_onsets) < 2:
            return 1.0

        # Compute inter-onset intervals
        iois = [analysis.note_onsets[i+1] - analysis.note_onsets[i]
               for i in range(len(analysis.note_onsets)-1)]

        if not iois:
            return 1.0

        # Lower std = more consistent
        std = np.std(iois)
        mean = np.mean(iois)

        if mean == 0:
            return 1.0

        cv = std / mean  # Coefficient of variation
        consistency = 1.0 - min(cv, 1.0)

        return float(np.clip(consistency, 0.0, 1.0))

    def _detect_polyrhythm(self, analysis: MIDIAnalysis) -> float:
        """Detect polyrhythmic complexity"""
        # Simplified: check if different tracks have different rhythmic patterns
        if len(analysis.tracks_by_instrument) < 2:
            return 0.0

        # Compare rhythmic entropy across tracks
        track_entropies = []
        for program, notes in analysis.tracks_by_instrument.items():
            durations = [n['duration'] for n in notes]
            track_entropies.append(self._compute_rhythm_entropy(durations))

        if len(track_entropies) < 2:
            return 0.0

        # Higher variance in entropies suggests polyrhythm
        entropy_variance = np.var(track_entropies)
        return float(min(entropy_variance * 2, 1.0))

    def _compute_swing_amount(self, analysis: MIDIAnalysis) -> float:
        """Measure swing ratio"""
        # Simplified: detect if eighth notes are played unevenly
        if not analysis.note_onsets:
            return 0.5  # Straight time

        beat_duration = 60.0 / analysis.tempo_bpm
        eighth_note = beat_duration / 2

        # Look for pairs of eighth notes
        swing_ratios = []
        for i in range(len(analysis.note_onsets) - 1):
            ioi = analysis.note_onsets[i+1] - analysis.note_onsets[i]
            if 0.3 * eighth_note < ioi < 1.2 * eighth_note:
                # Check if this could be a swing pair
                if i + 2 < len(analysis.note_onsets):
                    next_ioi = analysis.note_onsets[i+2] - analysis.note_onsets[i+1]
                    if 0.3 * eighth_note < next_ioi < 1.2 * eighth_note:
                        ratio = ioi / (ioi + next_ioi)
                        swing_ratios.append(ratio)

        if swing_ratios:
            return float(np.clip(np.median(swing_ratios), 0.5, 0.75))
        else:
            return 0.5  # Straight time

    def _extract_dynamics(self, analysis: MIDIAnalysis) -> Dict[str, float]:
        """Extract 2 dynamics parameters"""
        dynamics = {}

        if analysis.all_velocities:
            # overall_level: mean velocity
            dynamics['overall_level'] = float(np.mean(analysis.all_velocities) / 127.0)

            # range: velocity std dev
            dynamics['range'] = float(np.std(analysis.all_velocities) / 127.0)
        else:
            dynamics['overall_level'] = 0.5
            dynamics['range'] = 0.0

        return dynamics

    def _extract_texture(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract 2 texture parameters"""
        texture = {}

        # polyphony: max simultaneous notes
        if analysis.all_notes:
            max_poly = 0
            time_points = np.linspace(0, analysis.duration_seconds, int(analysis.duration_seconds * 10))

            for t in time_points:
                active_count = sum(1 for n in analysis.all_notes
                                 if n['onset'] <= t <= n['onset'] + n['duration'])
                max_poly = max(max_poly, active_count)

            texture['polyphony'] = int(max_poly)
        else:
            texture['polyphony'] = 0

        # density: notes per second
        if analysis.duration_seconds > 0:
            texture['density'] = float(analysis.total_notes / analysis.duration_seconds)
        else:
            texture['density'] = 0.0

        return texture

    # ========================================================================
    # LEVEL 3: Genre-Specific Details (22 parameters)
    # ========================================================================

    def _extract_level3(self, analysis: MIDIAnalysis, level1: Dict, level2: Dict) -> Dict[str, Any]:
        """Extract Level 3 genre-specific parameters"""
        level3 = {}

        # Universal orchestration (always active)
        level3['orchestration'] = {
            'instrument_count': len(analysis.instrument_programs),
            'register_balance': self._compute_register_balance(analysis),
            'legato_ratio': self._compute_legato_ratio(analysis),
            'section_contrast': 0.5,  # Would need section detection
            'repetition_level': 0.5    # Would need full form analysis
        }

        # Genre-specific parameters
        genre = level1['genre.primary']

        if genre == 'jazz':
            level3['jazz'] = self._extract_jazz_specific(analysis, level2)
        elif genre == 'classical':
            level3['classical'] = self._extract_classical_specific(analysis, level2)
        elif genre == 'rock':
            level3['rock'] = self._extract_rock_specific(analysis, level2)
        elif genre == 'electronic':
            level3['electronic'] = self._extract_electronic_specific(analysis, level2)
        elif genre == 'hiphop':
            level3['hiphop'] = self._extract_hiphop_specific(analysis, level2)
        elif genre == 'latin':
            level3['latin'] = self._extract_latin_specific(analysis, level2)

        return level3

    def _compute_register_balance(self, analysis: MIDIAnalysis) -> float:
        """Compute pitch register balance"""
        if not analysis.all_pitches:
            return 0.5

        middle_c = 60
        low_notes = sum(1 for p in analysis.all_pitches if p < middle_c)
        high_notes = sum(1 for p in analysis.all_pitches if p >= middle_c)

        total = low_notes + high_notes
        if total == 0:
            return 0.5

        # 0.0 = all low, 1.0 = all high, 0.5 = balanced
        return float(high_notes / total)

    def _compute_legato_ratio(self, analysis: MIDIAnalysis) -> float:
        """Compute articulation legato ratio"""
        if not analysis.all_notes or len(analysis.all_notes) < 2:
            return 0.9

        sorted_notes = sorted(analysis.all_notes, key=lambda n: n['onset'])
        legato_ratios = []

        for i in range(len(sorted_notes) - 1):
            duration = sorted_notes[i]['duration']
            ioi = sorted_notes[i+1]['onset'] - sorted_notes[i]['onset']

            if ioi > 0:
                ratio = min(duration / ioi, 1.0)
                legato_ratios.append(ratio)

        return float(np.mean(legato_ratios)) if legato_ratios else 0.9

    def _extract_jazz_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract jazz-specific parameters"""
        return {
            'swing_feel': self._categorize_swing(level2['rhythm']['swing_amount']),
            'walking_bass': self._detect_walking_bass(analysis),
            'improvisation_ratio': 0.3,  # Would need thematic analysis
            'bebop_vocabulary': 0.2       # Would need pattern matching
        }

    def _categorize_swing(self, swing_amount: float) -> str:
        """Categorize swing amount"""
        if swing_amount < 0.55:
            return 'straight'
        elif swing_amount < 0.62:
            return 'light'
        elif swing_amount < 0.72:
            return 'medium'
        else:
            return 'hard'

    def _detect_walking_bass(self, analysis: MIDIAnalysis) -> float:
        """Detect walking bass pattern"""
        # Look for bass program (32-39)
        bass_notes = []
        for program, notes in analysis.tracks_by_instrument.items():
            if 32 <= program <= 39:
                bass_notes.extend(notes)

        if not bass_notes:
            return 0.0

        # Check for quarter note patterns
        sorted_bass = sorted(bass_notes, key=lambda n: n['onset'])
        beat_duration = 60.0 / analysis.tempo_bpm

        quarter_note_count = 0
        for i in range(len(sorted_bass) - 1):
            ioi = sorted_bass[i+1]['onset'] - sorted_bass[i]['onset']
            if 0.9 * beat_duration < ioi < 1.1 * beat_duration:
                quarter_note_count += 1

        ratio = quarter_note_count / len(sorted_bass) if sorted_bass else 0.0
        return float(min(ratio, 1.0))

    def _extract_classical_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract classical-specific parameters"""
        return {
            'counterpoint': self._estimate_counterpoint(analysis),
            'development_density': 0.5,  # Would need motif tracking
            'voice_leading_quality': 0.7  # Would need voice leading analysis
        }

    def _estimate_counterpoint(self, analysis: MIDIAnalysis) -> float:
        """Estimate contrapuntal complexity"""
        if len(analysis.tracks_by_instrument) < 2:
            return 0.0

        # Check for independent rhythmic activity
        track_rhythms = {}
        for program, notes in analysis.tracks_by_instrument.items():
            durations = [n['duration'] for n in notes]
            track_rhythms[program] = self._compute_rhythm_entropy(durations)

        if len(track_rhythms) < 2:
            return 0.0

        # High variance = more independence
        rhythm_variance = np.var(list(track_rhythms.values()))
        return float(min(rhythm_variance * 3, 1.0))

    def _extract_rock_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract rock-specific parameters"""
        return {
            'power_chord_ratio': self._detect_power_chords(analysis),
            'riff_repetition': level2['melody']['repetition'],  # Use melody repetition as proxy
            'distortion_level': min(level2['dynamics']['overall_level'] * 1.2, 1.0)
        }

    def _detect_power_chords(self, analysis: MIDIAnalysis) -> float:
        """Detect power chord usage"""
        if not analysis.chord_notes:
            return 0.0

        power_chord_count = 0
        for chord in analysis.chord_notes:
            if len(chord) == 2:
                pitches = sorted([n['pitch'] for n in chord])
                interval = (pitches[1] - pitches[0]) % 12
                if interval == 7:  # Perfect fifth
                    power_chord_count += 1

        return float(power_chord_count / len(analysis.chord_notes))

    def _extract_electronic_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract electronic-specific parameters"""
        return {
            'quantization': level2['rhythm']['groove_consistency'],  # High consistency = high quantization
            'filter_movement': 0.5,  # Would need CC analysis
            'arpeggio_density': self._detect_arpeggios(analysis)
        }

    def _detect_arpeggios(self, analysis: MIDIAnalysis) -> float:
        """Detect arpeggio patterns"""
        # Look for ascending/descending note sequences
        if not analysis.melody_pitches or len(analysis.melody_pitches) < 3:
            return 0.0

        arpeggio_count = 0
        total_segments = len(analysis.melody_pitches) - 2

        for i in range(len(analysis.melody_pitches) - 2):
            # Check for consistent direction
            diff1 = analysis.melody_pitches[i+1] - analysis.melody_pitches[i]
            diff2 = analysis.melody_pitches[i+2] - analysis.melody_pitches[i+1]

            # Same direction and interval size (3-7 semitones = chord tones)
            if (diff1 > 0 and diff2 > 0) or (diff1 < 0 and diff2 < 0):
                if 3 <= abs(diff1) <= 7 and 3 <= abs(diff2) <= 7:
                    arpeggio_count += 1

        return float(arpeggio_count / total_segments) if total_segments > 0 else 0.0

    def _extract_hiphop_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract hip-hop-specific parameters"""
        return {
            'sample_based': level2['melody']['repetition'],  # High repetition = sample-based
            'boom_bap_feel': 0.5  # Would need specific drum pattern detection
        }

    def _extract_latin_specific(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract Latin-specific parameters"""
        return {
            'clave_pattern': self._detect_clave_pattern(analysis),
            'montuno_complexity': 0.5  # Would need piano pattern analysis
        }

    def _detect_clave_pattern(self, analysis: MIDIAnalysis) -> str:
        """Detect clave pattern type"""
        # Simplified clave detection - would need sophisticated rhythm analysis
        syncopation = analysis.note_onsets

        # Default to most common
        return "son_clave_2-3"


def main():
    """Test the extractor"""
    extractor = HierarchicalParameterExtractor(verbose=True)

    # Test with a sample MIDI file if available
    import sys
    if len(sys.argv) > 1:
        midi_path = sys.argv[1]
        params = extractor.extract_from_midi(midi_path)

        print("\n" + "="*80)
        print("HIERARCHICAL PARAMETER EXTRACTION RESULTS")
        print("="*80)

        print("\nLevel 1 - Global Context:")
        for key, value in params['level1_global'].items():
            print(f"  {key:30s} = {value}")

        print("\nLevel 2 - Universal Dimensions:")
        for category, subparams in params['level2_universal'].items():
            print(f"\n  {category.upper()}:")
            for key, value in subparams.items():
                print(f"    {key:28s} = {value}")

        print("\nLevel 3 - Genre-Specific Details:")
        for category, subparams in params['level3_genre_specific'].items():
            print(f"\n  {category.upper()}:")
            if isinstance(subparams, dict):
                for key, value in subparams.items():
                    print(f"    {key:28s} = {value}")

        print("\n" + "="*80)

        # Save to JSON
        output_path = midi_path.replace('.mid', '_hierarchical_params.json')
        with open(output_path, 'w') as f:
            json.dump(params, f, indent=2)
        print(f"\nParameters saved to: {output_path}")
    else:
        print("Usage: python hierarchical_extractor.py <midi_file.mid>")


if __name__ == "__main__":
    main()
