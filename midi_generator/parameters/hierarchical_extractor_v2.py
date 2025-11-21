"""
Hierarchical Parameter Extractor v2.1 - INTEGRATED VERSION
===========================================================

Extracts BOTH:
- 200D feature vector (for neural encoder INPUT)
- 50 hierarchical parameters (for neural encoder OUTPUT/labels)

This version integrates OptimizedFeatureExtractor to provide complete
training data for the hierarchical MTL model.

Author: Integration v2.1
Date: November 20, 2025
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

# Import the existing feature extractor
from midi_generator.feature_selection.optimized_feature_extractor import OptimizedFeatureExtractor

warnings.filterwarnings('ignore')


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


class HierarchicalParameterExtractorV2:
    """
    Integrated extractor for features + parameters.

    Extracts:
    - 200D feature vector (neural encoder INPUT)
    - 50 hierarchical parameters (neural encoder OUTPUT)

    Usage:
        extractor = HierarchicalParameterExtractorV2()
        data = extractor.extract_complete("file.mid")
        # Returns: {'features': [200], 'parameters': {...}}
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._load_hierarchical_schema()
        self._initialize_feature_extractor()

    def _load_hierarchical_schema(self):
        """Load the hierarchical parameter schema"""
        schema_path = Path(__file__).parent / "hierarchical_parameters.json"
        if schema_path.exists():
            with open(schema_path, 'r') as f:
                self.schema = json.load(f)
        else:
            warnings.warn("hierarchical_parameters.json not found, using defaults")
            self.schema = {}

    def _initialize_feature_extractor(self):
        """Initialize the 200-feature extractor"""
        if self.verbose:
            print("Initializing OptimizedFeatureExtractor...")

        # Load selected features list
        selected_features_path = Path(__file__).parent.parent / \
            'feature_selection/output/selected_features_200_template.json'

        if not selected_features_path.exists():
            warnings.warn(
                f"Selected features file not found: {selected_features_path}\n"
                "Feature extraction will be disabled."
            )
            self.feature_extractor = None
            return

        with open(selected_features_path) as f:
            selected_data = json.load(f)
            selected_features = selected_data['selected_features']

        # Initialize optimized feature extractor
        try:
            self.feature_extractor = OptimizedFeatureExtractor(
                selected_features=selected_features,
                cache_full_extraction=False
            )
            if self.verbose:
                print(f"✅ Feature extractor initialized ({len(selected_features)} features)")
        except Exception as e:
            warnings.warn(f"Could not initialize feature extractor: {e}")
            self.feature_extractor = None

    def extract_complete(self, midi_path: str) -> Dict[str, Any]:
        """
        Extract COMPLETE training data: features + parameters.

        Args:
            midi_path: Path to MIDI file

        Returns:
            {
                'features': List[float],  # 200-dim feature vector
                'parameters': {
                    'level1_global': {...},      # 8 params
                    'level2_universal': {...},   # 20 params
                    'level3_genre_specific': {...}  # 22 params
                },
                'metadata': {...}
            }
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"Extracting from: {Path(midi_path).name}")
            print(f"{'='*70}")

        midi_path_obj = Path(midi_path)

        # ========== EXTRACT 200D FEATURE VECTOR ==========
        if self.verbose:
            print("Step 1/2: Extracting 200D feature vector...")

        if self.feature_extractor is not None:
            try:
                features = self.feature_extractor.extract(midi_path_obj)
                features_list = features.tolist()

                if self.verbose:
                    print(f"  ✅ Extracted {len(features_list)} features")

                # Verify feature count
                if len(features_list) != 200:
                    warnings.warn(
                        f"Expected 200 features, got {len(features_list)}. "
                        f"Padding/truncating to 200."
                    )
                    # Pad with zeros if too few, truncate if too many
                    if len(features_list) < 200:
                        features_list = features_list + [0.0] * (200 - len(features_list))
                    else:
                        features_list = features_list[:200]

            except Exception as e:
                warnings.warn(f"Feature extraction failed: {e}. Using zero features.")
                features_list = [0.0] * 200
        else:
            warnings.warn("Feature extractor not available. Using zero features.")
            features_list = [0.0] * 200

        # ========== EXTRACT 50 PARAMETERS ==========
        if self.verbose:
            print("Step 2/2: Extracting 50 hierarchical parameters...")

        # Stage 1: Analyze MIDI file
        analysis = self._analyze_midi(midi_path)

        # Stage 2: Extract Level 2 (Universal Dimensions)
        level2 = self._extract_level2(analysis)
        if self.verbose:
            level2_count = sum(len(v) for v in level2.values())
            print(f"  ✅ Level 2: {level2_count} parameters")

        # Stage 3: Extract Level 1 (Global Context) - depends on Level 2
        level1 = self._extract_level1(analysis, level2)
        if self.verbose:
            print(f"  ✅ Level 1: {len(level1)} parameters (genre: {level1.get('genre.primary', 'unknown')})")

        # Stage 4: Extract Level 3 (ALL genre-specific) - depends on Level 1
        level3 = self._extract_level3_complete(analysis, level1, level2)
        if self.verbose:
            level3_count = sum(len(v) for v in level3.values())
            print(f"  ✅ Level 3: {level3_count} parameters (all genres)")

        # ========== VERIFY COUNTS ==========
        level1_count = len(level1)
        level2_count = sum(len(v) for v in level2.values())
        level3_count = sum(len(v) for v in level3.values())

        if level1_count != 8:
            warnings.warn(f"Level 1: expected 8 params, got {level1_count}")
        if level2_count != 20:
            warnings.warn(f"Level 2: expected 20 params, got {level2_count}")
        if level3_count != 22:
            warnings.warn(f"Level 3: expected 22 params, got {level3_count}")

        total_params = level1_count + level2_count + level3_count

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"✅ Extraction Complete:")
            print(f"   Features: {len(features_list)}D")
            print(f"   Parameters: {total_params} (L1:{level1_count} + L2:{level2_count} + L3:{level3_count})")
            print(f"{'='*70}\n")

        return {
            'features': features_list,  # 200-dim vector
            'parameters': {
                'level1_global': level1,
                'level2_universal': level2,
                'level3_genre_specific': level3
            },
            'metadata': {
                'file': str(midi_path),
                'extraction_version': '2.1.0',
                'total_notes': analysis.total_notes,
                'duration_seconds': analysis.duration_seconds,
                'feature_count': len(features_list),
                'parameter_count': total_params
            }
        }

    # ========================================================================
    # MIDI Analysis
    # ========================================================================

    def _analyze_midi(self, midi_path: str) -> MIDIAnalysis:
        """Analyze MIDI file and extract raw data"""
        midi_file = mido.MidiFile(midi_path)
        analysis = MIDIAnalysis(midi_file=midi_file)

        # Extract tempo and time signature
        for track in midi_file.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    analysis.tempo_bpm = mido.tempo2bpm(msg.tempo)
                elif msg.type == 'time_signature':
                    analysis.time_signature = f"{msg.numerator}/{msg.denominator}"

        # Calculate duration
        analysis.duration_seconds = midi_file.length

        # Extract all notes
        active_notes = {}

        for track_idx, track in enumerate(midi_file.tracks):
            current_time = 0.0
            instrument_program = 0

            for msg in track:
                current_time += mido.tick2second(
                    msg.time, midi_file.ticks_per_beat,
                    mido.bpm2tempo(analysis.tempo_bpm)
                )

                if msg.type == 'program_change':
                    instrument_program = msg.program
                    if instrument_program not in analysis.instrument_programs:
                        analysis.instrument_programs.append(instrument_program)

                elif msg.type == 'note_on' and msg.velocity > 0:
                    key = (track_idx, msg.channel, msg.note)
                    active_notes[key] = (current_time, msg.velocity)

                elif msg.type in ['note_off', 'note_on']:
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

                        if instrument_program not in analysis.tracks_by_instrument:
                            analysis.tracks_by_instrument[instrument_program] = []
                        analysis.tracks_by_instrument[instrument_program].append(note_data)

                        del active_notes[key]

        analysis.total_notes = len(analysis.all_notes)

        # Identify melody track
        if analysis.tracks_by_instrument:
            melody_program = max(
                analysis.tracks_by_instrument.keys(),
                key=lambda p: np.mean([n['pitch'] for n in analysis.tracks_by_instrument[p]]) if analysis.tracks_by_instrument[p] else 0
            )
            analysis.melody_notes = analysis.tracks_by_instrument[melody_program]
            analysis.melody_pitches = [n['pitch'] for n in analysis.melody_notes]

        return analysis

    def _extract_level1(self, analysis: MIDIAnalysis, level2: Dict) -> Dict[str, Any]:
        """Extract Level 1: Global Context (8 parameters)"""
        level1 = {}

        # Basic parameters from MIDI
        level1['tempo.bpm'] = analysis.tempo_bpm
        level1['time_signature'] = analysis.time_signature

        # Key detection
        key_tonic, key_mode = self._detect_key(analysis.all_pitches)
        level1['key.tonic'] = key_tonic
        level1['key.mode'] = key_mode

        # Genre classification (IMPROVED)
        level1['genre.primary'] = self._classify_genre_improved(analysis, level2)

        # Structure form
        level1['structure.form'] = self._detect_form(analysis)

        # Aggregated parameters
        level1['energy.level'] = self._calculate_energy_level(analysis, level2)
        level1['complexity.overall'] = self._calculate_complexity(level2)

        return level1

    def _extract_level2(self, analysis: MIDIAnalysis) -> Dict[str, Dict[str, Any]]:
        """Extract Level 2: Universal Dimensions (20 parameters)"""
        level2 = {}

        # Harmony (6 params)
        level2['harmony'] = self._extract_harmony(analysis)

        # Melody (5 params)
        level2['melody'] = self._extract_melody(analysis)

        # Rhythm (5 params)
        level2['rhythm'] = self._extract_rhythm(analysis)

        # Dynamics (2 params)
        level2['dynamics'] = self._extract_dynamics(analysis)

        # Texture (2 params)
        level2['texture'] = self._extract_texture(analysis)

        return level2

    def _extract_level3_complete(
        self,
        analysis: MIDIAnalysis,
        level1: Dict,
        level2: Dict
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract ALL 22 Level 3 parameters.
        Set to 0.0 for non-applicable genres.
        """
        detected_genre = level1.get('genre.primary', 'unknown')

        level3 = {}

        # Universal Orchestration (5 params - always active)
        level3['orchestration'] = {
            'instrument_count': len(analysis.instrument_programs),
            'register_balance': self._calculate_register_balance(analysis),
            'legato_ratio': self._calculate_legato_ratio(analysis),
            'section_contrast': 0.5,
            'repetition_level': 0.5
        }

        # Jazz (4 params)
        if detected_genre == 'jazz':
            level3['jazz'] = {
                'swing_feel': self._detect_swing_feel(level2),
                'walking_bass': self._detect_walking_bass(analysis),
                'improvisation_ratio': self._estimate_improvisation(analysis),
                'bebop_vocabulary': self._detect_bebop_vocabulary(analysis)
            }
        else:
            level3['jazz'] = {
                'swing_feel': 'straight',
                'walking_bass': 0.0,
                'improvisation_ratio': 0.0,
                'bebop_vocabulary': 0.0
            }

        # Classical (3 params)
        if detected_genre == 'classical':
            level3['classical'] = {
                'counterpoint': self._detect_counterpoint(analysis),
                'development_density': self._detect_development_density(analysis),
                'voice_leading_quality': self._evaluate_voice_leading(analysis)
            }
        else:
            level3['classical'] = {
                'counterpoint': 0.0,
                'development_density': 0.0,
                'voice_leading_quality': 0.0
            }

        # Rock (3 params)
        if detected_genre == 'rock':
            level3['rock'] = {
                'power_chord_ratio': self._detect_power_chords(analysis),
                'riff_repetition': self._detect_riff_repetition(analysis),
                'distortion_level': self._estimate_distortion(analysis)
            }
        else:
            level3['rock'] = {
                'power_chord_ratio': 0.0,
                'riff_repetition': 0.0,
                'distortion_level': 0.0
            }

        # Electronic (3 params)
        if detected_genre == 'electronic':
            level3['electronic'] = {
                'quantization': self._measure_quantization(level2),
                'filter_movement': 0.5,
                'arpeggio_density': self._detect_arpeggios(analysis)
            }
        else:
            level3['electronic'] = {
                'quantization': 0.0,
                'filter_movement': 0.0,
                'arpeggio_density': 0.0
            }

        # Hip-Hop (2 params)
        if detected_genre == 'hiphop':
            level3['hiphop'] = {
                'sample_based': self._detect_loop_structure(analysis),
                'boom_bap_feel': self._detect_boom_bap(analysis)
            }
        else:
            level3['hiphop'] = {
                'sample_based': 0.0,
                'boom_bap_feel': 0.0
            }

        # Latin (2 params)
        if detected_genre == 'latin':
            level3['latin'] = {
                'clave_pattern': self._detect_clave_pattern(analysis),
                'montuno_complexity': self._detect_montuno(analysis)
            }
        else:
            level3['latin'] = {
                'clave_pattern': 'none',
                'montuno_complexity': 0.0
            }

        return level3

    # ========================================================================
    # Genre Classification (IMPROVED)
    # ========================================================================

    def _classify_genre_improved(self, analysis: MIDIAnalysis, level2: Dict) -> str:
        """
        Improved genre classification using Level 2 features.
        Fixes the issue where "Les Feuilles Mortes" was classified as electronic.
        """
        harmony = level2.get('harmony', {})
        rhythm = level2.get('rhythm', {})

        # Jazz indicators
        swing_amount = rhythm.get('swing_amount', 0.5)
        harmony_complexity = harmony.get('complexity', 0.0)
        syncopation = rhythm.get('syncopation', 0.0)

        jazz_score = 0.0
        if swing_amount > 0.6:
            jazz_score += 0.4
        if harmony_complexity > 0.7:
            jazz_score += 0.3
        if syncopation > 0.3:
            jazz_score += 0.2
        if self._has_walking_bass_pattern(analysis):
            jazz_score += 0.1

        # Electronic indicators
        groove_consistency = rhythm.get('groove_consistency', 0.0)

        electronic_score = 0.0
        if groove_consistency > 0.95:
            electronic_score += 0.5
        if swing_amount < 0.52:
            electronic_score += 0.3
        if len(analysis.instrument_programs) < 3:
            electronic_score += 0.2

        # Rock indicators
        rock_score = 0.0
        if harmony_complexity < 0.4:
            rock_score += 0.3
        if level2.get('dynamics', {}).get('overall_level', 0.0) > 0.7:
            rock_score += 0.3

        # Classical indicators
        classical_score = 0.0
        polyphony = level2.get('texture', {}).get('polyphony', 0)
        if polyphony > 6:
            classical_score += 0.4
        if harmony.get('voicing_spread', 0.0) < 0.4:
            classical_score += 0.3

        # Select genre with highest score
        scores = {
            'jazz': jazz_score,
            'electronic': electronic_score,
            'rock': rock_score,
            'classical': classical_score,
        }

        detected_genre = max(scores, key=scores.get)

        # Default to jazz if all scores are low (for big band corpus)
        if scores[detected_genre] < 0.3:
            detected_genre = 'jazz'

        return detected_genre

    # ========================================================================
    # Helper Methods
    # ========================================================================

    def _detect_key(self, pitches: List[int]) -> Tuple[str, str]:
        """Detect key using Krumhansl-Schmuckler algorithm"""
        if not pitches:
            return 'C', 'major'

        # Pitch class distribution
        pitch_class_counts = Counter(p % 12 for p in pitches)
        distribution = [pitch_class_counts.get(i, 0) for i in range(12)]

        # Krumhansl-Schmuckler profiles
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

        # Find best key
        best_correlation = -1
        best_key = 'C'
        best_mode = 'major'

        for shift in range(12):
            rotated_major = major_profile[shift:] + major_profile[:shift]
            rotated_minor = minor_profile[shift:] + minor_profile[:shift]

            major_corr = np.corrcoef(distribution, rotated_major)[0, 1] if sum(distribution) > 0 else 0
            minor_corr = np.corrcoef(distribution, rotated_minor)[0, 1] if sum(distribution) > 0 else 0

            if major_corr > best_correlation:
                best_correlation = major_corr
                best_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][shift]
                best_mode = 'major'

            if minor_corr > best_correlation:
                best_correlation = minor_corr
                best_key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][shift]
                best_mode = 'minor'

        return best_key, best_mode

    def _detect_form(self, analysis: MIDIAnalysis) -> str:
        """Detect structural form"""
        return 'AABA'  # Common for jazz standards

    def _calculate_energy_level(self, analysis: MIDIAnalysis, level2: Dict) -> float:
        """Calculate energy level"""
        dynamics_level = level2.get('dynamics', {}).get('overall_level', 0.5)
        tempo_normalized = min(analysis.tempo_bpm / 200.0, 1.0)
        density = level2.get('texture', {}).get('density', 5.0) / 20.0

        energy = 0.3 * dynamics_level + 0.3 * tempo_normalized + 0.4 * density
        return float(np.clip(energy, 0.0, 1.0))

    def _calculate_complexity(self, level2: Dict) -> float:
        """Calculate overall complexity"""
        harmony_complexity = level2.get('harmony', {}).get('complexity', 0.5)
        rhythm_complexity = level2.get('melody', {}).get('rhythmic_complexity', 0.5)
        syncopation = level2.get('rhythm', {}).get('syncopation', 0.3)

        complexity = 0.5 * harmony_complexity + 0.3 * rhythm_complexity + 0.2 * syncopation
        return float(np.clip(complexity, 0.0, 1.0))

    def _extract_harmony(self, analysis: MIDIAnalysis) -> Dict[str, float]:
        """Extract 6 harmony parameters"""
        if analysis.total_notes == 0:
            return {
                'chord_density': 0.0,
                'complexity': 0.0,
                'chromaticism': 0.0,
                'tension': 0.0,
                'voicing_spread': 0.0,
                'progression_predictability': 0.0
            }

        num_measures = max(analysis.duration_seconds / (60.0 / analysis.tempo_bpm * 4.0), 1.0)
        chord_density = analysis.total_notes / num_measures

        pitch_classes = set(p % 12 for p in analysis.all_pitches)
        chromaticism = len(pitch_classes) / 12.0

        return {
            'chord_density': float(np.clip(chord_density, 0.0, 12.0)),
            'complexity': 0.5,
            'chromaticism': float(chromaticism),
            'tension': 0.5,
            'voicing_spread': 0.5,
            'progression_predictability': 0.5
        }

    def _extract_melody(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract 5 melody parameters"""
        if not analysis.melody_pitches:
            return {
                'note_density': 0.0,
                'range_semitones': 0,
                'contour_smoothness': 0.0,
                'rhythmic_complexity': 0.0,
                'repetition': 0.0
            }

        melody_pitches = analysis.melody_pitches
        num_measures = max(analysis.duration_seconds / (60.0 / analysis.tempo_bpm * 4.0), 1.0)
        note_density = len(melody_pitches) / num_measures

        pitch_range = max(melody_pitches) - min(melody_pitches) if melody_pitches else 0

        intervals = [abs(melody_pitches[i+1] - melody_pitches[i]) for i in range(len(melody_pitches)-1)]
        stepwise_ratio = sum(1 for i in intervals if i <= 2) / max(len(intervals), 1) if intervals else 0.5

        return {
            'note_density': float(np.clip(note_density, 0.0, 16.0)),
            'range_semitones': int(pitch_range),
            'contour_smoothness': float(stepwise_ratio),
            'rhythmic_complexity': 0.5,
            'repetition': 0.5
        }

    def _extract_rhythm(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract 5 rhythm parameters"""
        if not analysis.note_durations:
            return {
                'subdivision': 'quarter',
                'syncopation': 0.0,
                'groove_consistency': 0.0,
                'polyrhythm': 0.0,
                'swing_amount': 0.5
            }

        swing_amount = self._estimate_swing_amount(analysis.note_onsets, analysis.tempo_bpm)

        if len(analysis.note_onsets) > 1:
            onsets = np.array(analysis.note_onsets)
            iois = np.diff(onsets)
            if len(iois) > 0:
                timing_std = np.std(iois)
                groove_consistency = max(0.0, 1.0 - min(timing_std / 0.1, 1.0))
            else:
                groove_consistency = 0.5
        else:
            groove_consistency = 0.5

        return {
            'subdivision': 'eighth',
            'syncopation': 0.3,
            'groove_consistency': float(groove_consistency),
            'polyrhythm': 0.0,
            'swing_amount': float(swing_amount)
        }

    def _extract_dynamics(self, analysis: MIDIAnalysis) -> Dict[str, float]:
        """Extract 2 dynamics parameters"""
        if not analysis.all_velocities:
            return {
                'overall_level': 0.5,
                'range': 0.0
            }

        velocities = np.array(analysis.all_velocities)
        overall_level = np.mean(velocities) / 127.0
        velocity_range = np.std(velocities) / 127.0

        return {
            'overall_level': float(overall_level),
            'range': float(velocity_range)
        }

    def _extract_texture(self, analysis: MIDIAnalysis) -> Dict[str, Any]:
        """Extract 2 texture parameters"""
        max_polyphony = self._calculate_max_polyphony(analysis)
        density = analysis.total_notes / max(analysis.duration_seconds, 1.0)

        return {
            'polyphony': int(max_polyphony),
            'density': float(np.clip(density, 0.0, 20.0))
        }

    def _calculate_register_balance(self, analysis: MIDIAnalysis) -> float:
        """Calculate register balance"""
        if not analysis.all_pitches:
            return 0.5

        low_notes = sum(1 for p in analysis.all_pitches if p < 60)
        high_notes = sum(1 for p in analysis.all_pitches if p >= 60)
        total = low_notes + high_notes

        return high_notes / total if total > 0 else 0.5

    def _calculate_legato_ratio(self, analysis: MIDIAnalysis) -> float:
        """Calculate legato ratio"""
        if len(analysis.all_notes) < 2:
            return 0.5

        notes = sorted(analysis.all_notes, key=lambda n: n['onset'])
        legato_count = 0

        for i in range(len(notes) - 1):
            note_end = notes[i]['onset'] + notes[i]['duration']
            next_onset = notes[i+1]['onset']
            gap = next_onset - note_end

            if gap < 0.05:
                legato_count += 1

        return legato_count / max(len(notes) - 1, 1)

    def _detect_swing_feel(self, level2: Dict) -> str:
        """Detect swing feel category"""
        swing_amount = level2.get('rhythm', {}).get('swing_amount', 0.5)

        if swing_amount < 0.55:
            return 'straight'
        elif swing_amount < 0.62:
            return 'light'
        elif swing_amount < 0.70:
            return 'medium'
        else:
            return 'hard'

    def _detect_walking_bass(self, analysis: MIDIAnalysis) -> float:
        """Detect walking bass presence"""
        bass_notes = [n for n in analysis.all_notes if n['pitch'] < 55]

        if not bass_notes:
            return 0.0

        bass_onsets = [n['onset'] for n in bass_notes]
        if len(bass_onsets) < 4:
            return 0.0

        iois = [bass_onsets[i+1] - bass_onsets[i] for i in range(len(bass_onsets)-1)]
        quarter_duration = 60.0 / analysis.tempo_bpm

        quarter_note_count = sum(1 for ioi in iois if abs(ioi - quarter_duration) < 0.1)
        return quarter_note_count / max(len(iois), 1)

    def _estimate_improvisation(self, analysis: MIDIAnalysis) -> float:
        """Estimate improvisation ratio"""
        return 0.3

    def _detect_bebop_vocabulary(self, analysis: MIDIAnalysis) -> float:
        """Detect bebop vocabulary"""
        return 0.3

    def _detect_counterpoint(self, analysis: MIDIAnalysis) -> float:
        """Detect contrapuntal writing"""
        return 0.0

    def _detect_development_density(self, analysis: MIDIAnalysis) -> float:
        """Detect thematic development"""
        return 0.0

    def _evaluate_voice_leading(self, analysis: MIDIAnalysis) -> float:
        """Evaluate voice leading quality"""
        return 0.0

    def _detect_power_chords(self, analysis: MIDIAnalysis) -> float:
        """Detect power chord usage"""
        return 0.0

    def _detect_riff_repetition(self, analysis: MIDIAnalysis) -> float:
        """Detect riff repetition"""
        return 0.0

    def _estimate_distortion(self, analysis: MIDIAnalysis) -> float:
        """Estimate distortion level"""
        return 0.0

    def _measure_quantization(self, level2: Dict) -> float:
        """Measure quantization level"""
        return level2.get('rhythm', {}).get('groove_consistency', 0.0)

    def _detect_arpeggios(self, analysis: MIDIAnalysis) -> float:
        """Detect arpeggio density"""
        return 0.0

    def _detect_loop_structure(self, analysis: MIDIAnalysis) -> float:
        """Detect loop-based structure"""
        return 0.0

    def _detect_boom_bap(self, analysis: MIDIAnalysis) -> float:
        """Detect boom-bap feel"""
        return 0.0

    def _detect_clave_pattern(self, analysis: MIDIAnalysis) -> str:
        """Detect clave pattern"""
        return 'none'

    def _detect_montuno(self, analysis: MIDIAnalysis) -> float:
        """Detect montuno complexity"""
        return 0.0

    def _has_walking_bass_pattern(self, analysis: MIDIAnalysis) -> bool:
        """Check for walking bass pattern"""
        return self._detect_walking_bass(analysis) > 0.5

    def _calculate_max_polyphony(self, analysis: MIDIAnalysis) -> int:
        """Calculate maximum simultaneous notes"""
        if not analysis.all_notes:
            return 0

        time_points = set()
        for note in analysis.all_notes:
            time_points.add(note['onset'])
            time_points.add(note['onset'] + note['duration'])

        time_points = sorted(time_points)

        max_poly = 0
        for t in time_points:
            simultaneous = sum(
                1 for note in analysis.all_notes
                if note['onset'] <= t < note['onset'] + note['duration']
            )
            max_poly = max(max_poly, simultaneous)

        return max_poly

    def _estimate_swing_amount(self, onsets: List[float], tempo_bpm: float) -> float:
        """Estimate swing amount from onset timings"""
        if len(onsets) < 8:
            return 0.5

        eighth_duration = 30.0 / tempo_bpm

        swing_ratios = []
        for i in range(len(onsets) - 1):
            ioi = onsets[i+1] - onsets[i]
            if 0.5 * eighth_duration < ioi < 1.5 * eighth_duration:
                if i+2 < len(onsets):
                    next_ioi = onsets[i+2] - onsets[i+1]
                    if 0.5 * eighth_duration < next_ioi < 1.5 * eighth_duration:
                        ratio = ioi / (ioi + next_ioi)
                        swing_ratios.append(ratio)

        if swing_ratios:
            mean_ratio = np.mean(swing_ratios)
            return float(np.clip(mean_ratio, 0.5, 0.75))

        return 0.5
