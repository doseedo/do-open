"""
Harmony Semantic Encoder - Agent 2
===================================

Specialized encoder for discovering 30 harmony parameters through semantic learning.

This module implements a harmony-specific semantic encoder that discovers interpretable
parameters across 6 harmonic dimensions:

1. Chord Types (5 params): Major, minor, extended, altered chord frequencies
2. Voicings (5 params): Spread, close position, drop voicings, rootless
3. Progressions (5 params): ii-V-I, circle of fifths, chromatic, modal interchange
4. Voice Leading (5 params): Smoothness, contrary motion, crossing, resolution
5. Harmonic Rhythm (5 params): Change rate, anticipation, suspension, pedal tones
6. Extensions/Tension (5 params): 9th, 11th, 13th usage, tension resolution

Integration Points:
- Tymoczko geometric validation (neo-Riemannian transformations)
- Big band harmony agents (BrassArranger, VoiceLeadingOptimizer, SaxVoicing)
- Musical locality functions (TRANSPOSE, INVERT, OCTAVE_SHIFT, VOICE_PERMUTATION)

Architecture:
    Input: 200D features → Hidden: 512D → Output: 30D harmony parameters

Author: Agent 2 - Harmony Module Builder
Date: November 21, 2025
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from enum import Enum
import json
import warnings

# Core imports
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not available")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

# Import existing infrastructure
try:
    from midi_generator.learning.semantic_encoder import (
        SemanticFeatureEncoder,
        EncoderConfig
    )
    from midi_generator.learning.musical_locality import LocalityType
    from midi_generator.core.neo_riemannian import (
        NeoRiemannianTransformations,
        Triad,
        TriadQuality
    )
    INFRASTRUCTURE_AVAILABLE = True
except ImportError:
    INFRASTRUCTURE_AVAILABLE = False
    warnings.warn("Core infrastructure not available")

# Try to import big band harmony agents
try:
    from midi_generator.transformation.voice_leading_optimizer import VoiceLeadingOptimizer
    VOICE_LEADING_AVAILABLE = True
except ImportError:
    VOICE_LEADING_AVAILABLE = False

try:
    import mido
    from mido import MidiFile
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False
    warnings.warn("mido not available - MIDI analysis will be limited")


# ============================================================================
# Harmony Parameter Categories
# ============================================================================

class HarmonyParameterCategory(Enum):
    """Categories of harmony parameters"""
    CHORD_TYPES = "chord_types"
    VOICINGS = "voicings"
    PROGRESSIONS = "progressions"
    VOICE_LEADING = "voice_leading"
    HARMONIC_RHYTHM = "harmonic_rhythm"
    EXTENSIONS_TENSION = "extensions_tension"


@dataclass
class HarmonyParameters:
    """Container for 30 discovered harmony parameters"""

    # Chord Types (5 parameters)
    major_chord_frequency: float = 0.0
    minor_chord_frequency: float = 0.0
    dominant_7th_frequency: float = 0.0
    extended_chord_frequency: float = 0.0
    altered_chord_frequency: float = 0.0

    # Voicings (5 parameters)
    voicing_spread_preference: float = 0.0
    close_position_ratio: float = 0.0
    drop_2_usage: float = 0.0
    drop_3_usage: float = 0.0
    rootless_voicing_ratio: float = 0.0

    # Progressions (5 parameters)
    ii_V_I_frequency: float = 0.0
    circle_of_fifths_adherence: float = 0.0
    chromatic_movement: float = 0.0
    parallel_motion_ratio: float = 0.0
    modal_interchange_usage: float = 0.0

    # Voice Leading (5 parameters)
    voice_leading_smoothness: float = 0.0
    contrary_motion_preference: float = 0.0
    voice_crossing_frequency: float = 0.0
    half_step_resolution_ratio: float = 0.0
    tritone_resolution_adherence: float = 0.0

    # Harmonic Rhythm (5 parameters)
    chord_change_rate: float = 0.0
    harmonic_anticipation: float = 0.0
    harmonic_suspension: float = 0.0
    pedal_tone_usage: float = 0.0
    harmonic_density: float = 0.0

    # Extensions/Tension (5 parameters)
    ninth_usage: float = 0.0
    eleventh_usage: float = 0.0
    thirteenth_usage: float = 0.0
    tension_resolution_ratio: float = 0.0
    avoid_note_handling: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            # Chord Types
            'major_chord_frequency': self.major_chord_frequency,
            'minor_chord_frequency': self.minor_chord_frequency,
            'dominant_7th_frequency': self.dominant_7th_frequency,
            'extended_chord_frequency': self.extended_chord_frequency,
            'altered_chord_frequency': self.altered_chord_frequency,

            # Voicings
            'voicing_spread_preference': self.voicing_spread_preference,
            'close_position_ratio': self.close_position_ratio,
            'drop_2_usage': self.drop_2_usage,
            'drop_3_usage': self.drop_3_usage,
            'rootless_voicing_ratio': self.rootless_voicing_ratio,

            # Progressions
            'ii_V_I_frequency': self.ii_V_I_frequency,
            'circle_of_fifths_adherence': self.circle_of_fifths_adherence,
            'chromatic_movement': self.chromatic_movement,
            'parallel_motion_ratio': self.parallel_motion_ratio,
            'modal_interchange_usage': self.modal_interchange_usage,

            # Voice Leading
            'voice_leading_smoothness': self.voice_leading_smoothness,
            'contrary_motion_preference': self.contrary_motion_preference,
            'voice_crossing_frequency': self.voice_crossing_frequency,
            'half_step_resolution_ratio': self.half_step_resolution_ratio,
            'tritone_resolution_adherence': self.tritone_resolution_adherence,

            # Harmonic Rhythm
            'chord_change_rate': self.chord_change_rate,
            'harmonic_anticipation': self.harmonic_anticipation,
            'harmonic_suspension': self.harmonic_suspension,
            'pedal_tone_usage': self.pedal_tone_usage,
            'harmonic_density': self.harmonic_density,

            # Extensions/Tension
            'ninth_usage': self.ninth_usage,
            'eleventh_usage': self.eleventh_usage,
            'thirteenth_usage': self.thirteenth_usage,
            'tension_resolution_ratio': self.tension_resolution_ratio,
            'avoid_note_handling': self.avoid_note_handling,
        }

    def to_vector(self) -> np.ndarray:
        """Convert to numpy vector (30D)"""
        values = [
            # Chord Types (5)
            self.major_chord_frequency,
            self.minor_chord_frequency,
            self.dominant_7th_frequency,
            self.extended_chord_frequency,
            self.altered_chord_frequency,

            # Voicings (5)
            self.voicing_spread_preference,
            self.close_position_ratio,
            self.drop_2_usage,
            self.drop_3_usage,
            self.rootless_voicing_ratio,

            # Progressions (5)
            self.ii_V_I_frequency,
            self.circle_of_fifths_adherence,
            self.chromatic_movement,
            self.parallel_motion_ratio,
            self.modal_interchange_usage,

            # Voice Leading (5)
            self.voice_leading_smoothness,
            self.contrary_motion_preference,
            self.voice_crossing_frequency,
            self.half_step_resolution_ratio,
            self.tritone_resolution_adherence,

            # Harmonic Rhythm (5)
            self.chord_change_rate,
            self.harmonic_anticipation,
            self.harmonic_suspension,
            self.pedal_tone_usage,
            self.harmonic_density,

            # Extensions/Tension (5)
            self.ninth_usage,
            self.eleventh_usage,
            self.thirteenth_usage,
            self.tension_resolution_ratio,
            self.avoid_note_handling,
        ]
        return np.array(values, dtype=np.float32)


# ============================================================================
# Chord Analysis
# ============================================================================

@dataclass
class ChordInfo:
    """Information about a detected chord"""
    onset_time: float
    duration: float
    pitch_classes: List[int]
    root: Optional[int] = None
    quality: Optional[str] = None
    voicing_spread: float = 0.0
    bass_note: Optional[int] = None
    extensions: List[int] = field(default_factory=list)

    def is_major(self) -> bool:
        """Check if chord is major"""
        return self.quality and 'major' in self.quality.lower()

    def is_minor(self) -> bool:
        """Check if chord is minor"""
        return self.quality and 'minor' in self.quality.lower()

    def has_seventh(self) -> bool:
        """Check if chord has a seventh"""
        return len(self.extensions) > 0 or (self.quality and '7' in self.quality)

    def is_extended(self) -> bool:
        """Check if chord has extensions beyond 7th"""
        return any(ext in [9, 11, 13] for ext in self.extensions)


class ChordAnalyzer:
    """
    Analyzes chords from MIDI data.

    Detects:
    - Chord types and qualities
    - Voicings and spacing
    - Extensions and alterations
    - Bass notes and inversions
    """

    def __init__(self, time_window: float = 0.05):
        """
        Initialize chord analyzer.

        Args:
            time_window: Time window (seconds) for grouping simultaneous notes
        """
        self.time_window = time_window

    def analyze_chords(self, midi_file: str) -> List[ChordInfo]:
        """
        Analyze chords in MIDI file.

        Args:
            midi_file: Path to MIDI file

        Returns:
            List of detected chords
        """
        if not MIDO_AVAILABLE:
            warnings.warn("mido not available - cannot analyze chords")
            return []

        try:
            mid = MidiFile(midi_file)
        except Exception as e:
            warnings.warn(f"Could not load MIDI file: {e}")
            return []

        # Extract note events
        notes = self._extract_notes(mid)

        # Group into chords
        chords = self._group_into_chords(notes)

        # Analyze each chord
        analyzed_chords = []
        for chord_notes in chords:
            chord_info = self._analyze_chord(chord_notes)
            analyzed_chords.append(chord_info)

        return analyzed_chords

    def _extract_notes(self, mid: MidiFile) -> List[Dict]:
        """Extract note events from MIDI file"""
        notes = []
        current_time = 0.0
        tempo = 500000  # Default tempo (120 BPM)

        for track in mid.tracks:
            track_time = 0.0
            active_notes = {}

            for msg in track:
                track_time += mido.tick2second(msg.time, mid.ticks_per_beat, tempo)

                if msg.type == 'set_tempo':
                    tempo = msg.tempo
                elif msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = {
                        'onset': track_time,
                        'velocity': msg.velocity
                    }
                elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                    if msg.note in active_notes:
                        note_data = active_notes.pop(msg.note)
                        notes.append({
                            'pitch': msg.note,
                            'onset': note_data['onset'],
                            'offset': track_time,
                            'duration': track_time - note_data['onset'],
                            'velocity': note_data['velocity']
                        })

        return sorted(notes, key=lambda n: n['onset'])

    def _group_into_chords(self, notes: List[Dict]) -> List[List[Dict]]:
        """Group notes into chords based on onset times"""
        if not notes:
            return []

        chords = []
        current_chord = [notes[0]]
        current_onset = notes[0]['onset']

        for note in notes[1:]:
            if abs(note['onset'] - current_onset) <= self.time_window:
                current_chord.append(note)
            else:
                if len(current_chord) >= 3:  # At least 3 notes for a chord
                    chords.append(current_chord)
                current_chord = [note]
                current_onset = note['onset']

        # Add last chord
        if len(current_chord) >= 3:
            chords.append(current_chord)

        return chords

    def _analyze_chord(self, chord_notes: List[Dict]) -> ChordInfo:
        """Analyze a single chord"""
        if not chord_notes:
            return ChordInfo(onset_time=0, duration=0, pitch_classes=[])

        # Basic properties
        onset = chord_notes[0]['onset']
        duration = max(n['offset'] for n in chord_notes) - onset
        pitches = [n['pitch'] for n in chord_notes]
        pitch_classes = sorted(set(p % 12 for p in pitches))

        # Voicing spread
        voicing_spread = max(pitches) - min(pitches)

        # Bass note
        bass_note = min(pitches)

        # Detect root and quality
        root, quality = self._detect_chord_quality(pitch_classes)

        # Detect extensions
        extensions = self._detect_extensions(pitch_classes, root)

        return ChordInfo(
            onset_time=onset,
            duration=duration,
            pitch_classes=pitch_classes,
            root=root,
            quality=quality,
            voicing_spread=voicing_spread,
            bass_note=bass_note,
            extensions=extensions
        )

    def _detect_chord_quality(self, pitch_classes: List[int]) -> Tuple[Optional[int], Optional[str]]:
        """
        Detect chord root and quality from pitch classes.

        Returns:
            Tuple of (root, quality_string)
        """
        if len(pitch_classes) < 3:
            return None, None

        # Try each pitch class as potential root
        for root in pitch_classes:
            # Normalize intervals from root
            intervals = sorted(set((pc - root) % 12 for pc in pitch_classes))

            # Major triad: 0, 4, 7
            if 0 in intervals and 4 in intervals and 7 in intervals:
                if 10 in intervals:
                    return root, "dominant_7th"
                return root, "major"

            # Minor triad: 0, 3, 7
            if 0 in intervals and 3 in intervals and 7 in intervals:
                if 10 in intervals:
                    return root, "minor_7th"
                return root, "minor"

            # Diminished: 0, 3, 6
            if 0 in intervals and 3 in intervals and 6 in intervals:
                return root, "diminished"

            # Augmented: 0, 4, 8
            if 0 in intervals and 4 in intervals and 8 in intervals:
                return root, "augmented"

        return None, "unknown"

    def _detect_extensions(self, pitch_classes: List[int], root: Optional[int]) -> List[int]:
        """Detect chord extensions (9, 11, 13)"""
        if root is None:
            return []

        extensions = []
        intervals = set((pc - root) % 12 for pc in pitch_classes)

        # 9th = 2 semitones
        if 2 in intervals or 1 in intervals or 3 in intervals:
            extensions.append(9)

        # 11th = 5 semitones
        if 5 in intervals or 6 in intervals:
            extensions.append(11)

        # 13th = 9 semitones
        if 9 in intervals or 8 in intervals or 10 in intervals:
            extensions.append(13)

        return extensions


# ============================================================================
# Harmony Semantic Encoder
# ============================================================================

if TORCH_AVAILABLE and INFRASTRUCTURE_AVAILABLE:

    class HarmonySemanticEncoder(SemanticFeatureEncoder):
        """
        Harmony-specific semantic encoder discovering 30 harmony parameters.

        This encoder extends SemanticFeatureEncoder with:
        - Harmony-specific locality functions (transpose, invert, octave shift, voice permutation)
        - Tymoczko geometric validation using neo-Riemannian transformations
        - Integration with big band harmony agents
        - Chord, voicing, and progression analysis

        Architecture:
            Input: 200D musical features
            Hidden: 512D
            Output: 30D harmony parameters

        Usage:
            encoder = HarmonySemanticEncoder()

            # Extract harmony parameters from features
            features = torch.randn(1, 200)
            harmony_params = encoder.extract_semantic_features(features)

            # Analyze MIDI file
            params = encoder.analyze_midi("piece.mid")
        """

        def __init__(self, config: Optional[EncoderConfig] = None):
            """
            Initialize harmony encoder.

            Args:
                config: Optional custom configuration. If None, uses defaults for harmony module.
            """
            if config is None:
                config = EncoderConfig(
                    input_dim=200,
                    hidden_dim=512,
                    num_semantic_features=30,  # 30 harmony parameters
                    num_locality_types=4,  # TRANSPOSE, INVERT, OCTAVE_SHIFT, VOICE_PERMUTATION
                    feature_activation='relu',
                    normalize_features=True
                )

            super().__init__(config)

            # Harmony-specific components
            self.chord_analyzer = ChordAnalyzer()
            self.neo_riemannian = NeoRiemannianTransformations() if INFRASTRUCTURE_AVAILABLE else None
            self.voice_leading_optimizer = None

            # Try to load voice leading optimizer
            if VOICE_LEADING_AVAILABLE:
                try:
                    self.voice_leading_optimizer = VoiceLeadingOptimizer()
                except Exception as e:
                    warnings.warn(f"Could not initialize VoiceLeadingOptimizer: {e}")

            # Harmony locality functions
            self.harmony_locality_functions = [
                LocalityType.TRANSPOSE,
                LocalityType.INVERT,
                LocalityType.OCTAVE_SHIFT,
                LocalityType.VOICE_PERMUTATION
            ]

        def extract_harmony_parameters(
            self,
            semantic_features: torch.Tensor
        ) -> HarmonyParameters:
            """
            Extract interpretable harmony parameters from semantic features.

            Args:
                semantic_features: Semantic features [30]

            Returns:
                HarmonyParameters object with all 30 parameters
            """
            if isinstance(semantic_features, torch.Tensor):
                features = semantic_features.detach().cpu().numpy()
            else:
                features = np.array(semantic_features)

            # Ensure correct dimensionality
            if features.ndim > 1:
                features = features.squeeze()

            if len(features) != 30:
                raise ValueError(f"Expected 30 features, got {len(features)}")

            # Map features to harmony parameters
            # Each category gets 5 consecutive features
            params = HarmonyParameters(
                # Chord Types (0-4)
                major_chord_frequency=float(features[0]),
                minor_chord_frequency=float(features[1]),
                dominant_7th_frequency=float(features[2]),
                extended_chord_frequency=float(features[3]),
                altered_chord_frequency=float(features[4]),

                # Voicings (5-9)
                voicing_spread_preference=float(features[5]),
                close_position_ratio=float(features[6]),
                drop_2_usage=float(features[7]),
                drop_3_usage=float(features[8]),
                rootless_voicing_ratio=float(features[9]),

                # Progressions (10-14)
                ii_V_I_frequency=float(features[10]),
                circle_of_fifths_adherence=float(features[11]),
                chromatic_movement=float(features[12]),
                parallel_motion_ratio=float(features[13]),
                modal_interchange_usage=float(features[14]),

                # Voice Leading (15-19)
                voice_leading_smoothness=float(features[15]),
                contrary_motion_preference=float(features[16]),
                voice_crossing_frequency=float(features[17]),
                half_step_resolution_ratio=float(features[18]),
                tritone_resolution_adherence=float(features[19]),

                # Harmonic Rhythm (20-24)
                chord_change_rate=float(features[20]),
                harmonic_anticipation=float(features[21]),
                harmonic_suspension=float(features[22]),
                pedal_tone_usage=float(features[23]),
                harmonic_density=float(features[24]),

                # Extensions/Tension (25-29)
                ninth_usage=float(features[25]),
                eleventh_usage=float(features[26]),
                thirteenth_usage=float(features[27]),
                tension_resolution_ratio=float(features[28]),
                avoid_note_handling=float(features[29])
            )

            return params

        def analyze_midi(self, midi_path: str) -> Dict[str, Any]:
            """
            Analyze MIDI file and extract harmony parameters.

            This method:
            1. Extracts chords using ChordAnalyzer
            2. Computes harmonic statistics
            3. Validates using Tymoczko geometry (neo-Riemannian)
            4. Returns comprehensive harmony analysis

            Args:
                midi_path: Path to MIDI file

            Returns:
                Dictionary with harmony analysis and parameters
            """
            # Analyze chords
            chords = self.chord_analyzer.analyze_chords(midi_path)

            if not chords:
                warnings.warn(f"No chords detected in {midi_path}")
                return {
                    'num_chords': 0,
                    'parameters': HarmonyParameters().to_dict(),
                    'analysis': {}
                }

            # Compute statistics
            stats = self._compute_chord_statistics(chords)

            # Analyze progressions
            progressions = self._analyze_progressions(chords)

            # Validate with Tymoczko geometry
            geometric_validation = self._validate_with_tymoczko(chords)

            # Create harmony parameters from analysis
            params = self._create_parameters_from_analysis(stats, progressions, geometric_validation)

            return {
                'num_chords': len(chords),
                'parameters': params.to_dict(),
                'statistics': stats,
                'progressions': progressions,
                'geometric_validation': geometric_validation
            }

        def _compute_chord_statistics(self, chords: List[ChordInfo]) -> Dict[str, float]:
            """Compute statistical properties of chord sequence"""
            if not chords:
                return {}

            total = len(chords)

            # Chord type frequencies
            major_count = sum(1 for c in chords if c.is_major())
            minor_count = sum(1 for c in chords if c.is_minor())
            seventh_count = sum(1 for c in chords if c.has_seventh())
            extended_count = sum(1 for c in chords if c.is_extended())

            # Voicing statistics
            spreads = [c.voicing_spread for c in chords if c.voicing_spread > 0]
            avg_spread = np.mean(spreads) if spreads else 0.0
            close_position_count = sum(1 for s in spreads if s <= 12)  # Within octave

            # Harmonic rhythm
            durations = [c.duration for c in chords if c.duration > 0]
            avg_duration = np.mean(durations) if durations else 0.0

            return {
                'major_frequency': major_count / total,
                'minor_frequency': minor_count / total,
                'seventh_frequency': seventh_count / total,
                'extended_frequency': extended_count / total,
                'average_voicing_spread': avg_spread,
                'close_position_ratio': close_position_count / len(spreads) if spreads else 0.0,
                'average_chord_duration': avg_duration,
                'chord_change_rate': 1.0 / avg_duration if avg_duration > 0 else 0.0
            }

        def _analyze_progressions(self, chords: List[ChordInfo]) -> Dict[str, Any]:
            """Analyze chord progressions"""
            if len(chords) < 2:
                return {'ii_V_I_count': 0, 'chromatic_movement': 0.0}

            # Look for ii-V-I progressions
            ii_V_I_count = 0
            chromatic_moves = 0
            fifth_moves = 0

            for i in range(len(chords) - 2):
                c1, c2, c3 = chords[i], chords[i+1], chords[i+2]

                if all(c.root is not None for c in [c1, c2, c3]):
                    # Check for ii-V-I
                    interval1 = (c2.root - c1.root) % 12
                    interval2 = (c3.root - c2.root) % 12

                    if interval1 == 7 and interval2 == 7:  # Both up a fifth
                        if c1.is_minor() and 'dominant' in (c2.quality or ''):
                            ii_V_I_count += 1

            # Analyze adjacent chord movements
            for i in range(len(chords) - 1):
                c1, c2 = chords[i], chords[i+1]

                if c1.root is not None and c2.root is not None:
                    interval = (c2.root - c1.root) % 12

                    if interval in [1, 2, 10, 11]:  # Chromatic movement
                        chromatic_moves += 1
                    elif interval == 7 or interval == 5:  # Fifth movement
                        fifth_moves += 1

            total_moves = len(chords) - 1

            return {
                'ii_V_I_count': ii_V_I_count,
                'ii_V_I_frequency': ii_V_I_count / max(len(chords) - 2, 1),
                'chromatic_movement': chromatic_moves / total_moves if total_moves > 0 else 0.0,
                'fifth_movement': fifth_moves / total_moves if total_moves > 0 else 0.0
            }

        def _validate_with_tymoczko(self, chords: List[ChordInfo]) -> Dict[str, Any]:
            """
            Validate chord progressions using Tymoczko geometric theory.

            Uses neo-Riemannian transformations to analyze voice leading efficiency.
            """
            if not self.neo_riemannian or len(chords) < 2:
                return {'available': False}

            validation = {
                'available': True,
                'neo_riemannian_transformations': [],
                'smooth_voice_leading_ratio': 0.0
            }

            smooth_transitions = 0
            total_transitions = 0

            for i in range(len(chords) - 1):
                c1, c2 = chords[i], chords[i+1]

                # Try to create Triad objects
                if c1.root is not None and c2.root is not None:
                    try:
                        # Determine quality
                        quality1 = TriadQuality.MAJOR if c1.is_major() else TriadQuality.MINOR
                        quality2 = TriadQuality.MAJOR if c2.is_major() else TriadQuality.MINOR

                        triad1 = Triad(root=c1.root, quality=quality1)
                        triad2 = Triad(root=c2.root, quality=quality2)

                        # Check for neo-Riemannian transformations
                        if self.neo_riemannian.parallel(triad1) == triad2:
                            validation['neo_riemannian_transformations'].append({
                                'type': 'parallel',
                                'from': str(triad1),
                                'to': str(triad2)
                            })
                            smooth_transitions += 1
                        elif self.neo_riemannian.relative(triad1) == triad2:
                            validation['neo_riemannian_transformations'].append({
                                'type': 'relative',
                                'from': str(triad1),
                                'to': str(triad2)
                            })
                            smooth_transitions += 1
                        elif self.neo_riemannian.leading_tone(triad1) == triad2:
                            validation['neo_riemannian_transformations'].append({
                                'type': 'leading_tone',
                                'from': str(triad1),
                                'to': str(triad2)
                            })
                            smooth_transitions += 1

                        total_transitions += 1
                    except Exception as e:
                        warnings.warn(f"Error in neo-Riemannian analysis: {e}")

            if total_transitions > 0:
                validation['smooth_voice_leading_ratio'] = smooth_transitions / total_transitions

            return validation

        def _create_parameters_from_analysis(
            self,
            stats: Dict[str, float],
            progressions: Dict[str, Any],
            geometric: Dict[str, Any]
        ) -> HarmonyParameters:
            """Create HarmonyParameters from analysis results"""

            return HarmonyParameters(
                # Chord Types - from statistics
                major_chord_frequency=stats.get('major_frequency', 0.0),
                minor_chord_frequency=stats.get('minor_frequency', 0.0),
                dominant_7th_frequency=stats.get('seventh_frequency', 0.0),
                extended_chord_frequency=stats.get('extended_frequency', 0.0),
                altered_chord_frequency=0.0,  # Would need more sophisticated detection

                # Voicings - from statistics
                voicing_spread_preference=stats.get('average_voicing_spread', 0.0) / 48.0,  # Normalize
                close_position_ratio=stats.get('close_position_ratio', 0.0),
                drop_2_usage=0.0,  # Would need voicing-specific detection
                drop_3_usage=0.0,
                rootless_voicing_ratio=0.0,

                # Progressions - from progression analysis
                ii_V_I_frequency=progressions.get('ii_V_I_frequency', 0.0),
                circle_of_fifths_adherence=progressions.get('fifth_movement', 0.0),
                chromatic_movement=progressions.get('chromatic_movement', 0.0),
                parallel_motion_ratio=0.0,  # Would need voice-by-voice analysis
                modal_interchange_usage=0.0,  # Would need key detection

                # Voice Leading - from geometric validation
                voice_leading_smoothness=geometric.get('smooth_voice_leading_ratio', 0.0),
                contrary_motion_preference=0.0,  # Would need voice-by-voice analysis
                voice_crossing_frequency=0.0,
                half_step_resolution_ratio=0.0,
                tritone_resolution_adherence=0.0,

                # Harmonic Rhythm - from statistics
                chord_change_rate=stats.get('chord_change_rate', 0.0),
                harmonic_anticipation=0.0,
                harmonic_suspension=0.0,
                pedal_tone_usage=0.0,
                harmonic_density=stats.get('seventh_frequency', 0.0) + stats.get('extended_frequency', 0.0),

                # Extensions/Tension - would need detailed extension analysis
                ninth_usage=0.0,
                eleventh_usage=0.0,
                thirteenth_usage=0.0,
                tension_resolution_ratio=0.0,
                avoid_note_handling=0.0
            )

        def get_locality_functions(self) -> List[LocalityType]:
            """Get harmony-specific locality functions"""
            return self.harmony_locality_functions

        def save_analysis(self, analysis: Dict[str, Any], output_path: Path):
            """Save harmony analysis to JSON"""
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            print(f"✅ Harmony analysis saved to {output_path}")


# ============================================================================
# Utility Functions
# ============================================================================

def create_default_harmony_encoder() -> 'HarmonySemanticEncoder':
    """Create harmony encoder with default configuration"""
    if not TORCH_AVAILABLE or not INFRASTRUCTURE_AVAILABLE:
        raise RuntimeError("PyTorch and infrastructure required")

    return HarmonySemanticEncoder()


def analyze_harmony_corpus(
    corpus_dir: Path,
    output_dir: Path,
    max_files: Optional[int] = None
) -> Dict[str, Any]:
    """
    Analyze harmony in a corpus of MIDI files.

    Args:
        corpus_dir: Directory containing MIDI files
        output_dir: Output directory for results
        max_files: Maximum number of files to analyze

    Returns:
        Aggregated harmony statistics
    """
    encoder = create_default_harmony_encoder()

    # Find MIDI files
    midi_files = list(corpus_dir.glob("**/*.mid")) + list(corpus_dir.glob("**/*.midi"))

    if max_files:
        midi_files = midi_files[:max_files]

    print(f"Analyzing {len(midi_files)} MIDI files...")

    # Analyze each file
    results = []
    for i, midi_path in enumerate(midi_files):
        try:
            analysis = encoder.analyze_midi(str(midi_path))
            analysis['file'] = midi_path.name
            results.append(analysis)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(midi_files)} files")
        except Exception as e:
            warnings.warn(f"Error analyzing {midi_path.name}: {e}")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "harmony_corpus_analysis.json"

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Corpus analysis complete: {results_path}")

    # Aggregate statistics
    if results:
        aggregated = _aggregate_harmony_statistics(results)

        agg_path = output_dir / "harmony_aggregated_stats.json"
        with open(agg_path, 'w') as f:
            json.dump(aggregated, f, indent=2)

        print(f"✅ Aggregated statistics: {agg_path}")

        return aggregated

    return {}


def _aggregate_harmony_statistics(results: List[Dict]) -> Dict[str, Any]:
    """Aggregate harmony statistics across corpus"""

    # Collect all parameters
    all_params = []
    for result in results:
        if 'parameters' in result and result['parameters']:
            all_params.append(result['parameters'])

    if not all_params:
        return {}

    # Compute means and stds
    aggregated = {
        'num_files': len(results),
        'num_valid_analyses': len(all_params),
        'parameter_means': {},
        'parameter_stds': {}
    }

    # Get parameter names from first entry
    param_names = all_params[0].keys()

    for param in param_names:
        values = [p[param] for p in all_params if param in p and p[param] is not None]
        if values:
            aggregated['parameter_means'][param] = float(np.mean(values))
            aggregated['parameter_stds'][param] = float(np.std(values))

    return aggregated


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Test harmony encoder"""
    print("="*80)
    print("HARMONY SEMANTIC ENCODER - Agent 2")
    print("="*80)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch not available")
        return

    if not INFRASTRUCTURE_AVAILABLE:
        print("\n❌ Infrastructure not available")
        return

    # Create encoder
    print("\n1. Creating harmony encoder...")
    encoder = create_default_harmony_encoder()
    print(f"   ✅ Encoder created with 30 harmony parameters")

    # Test forward pass
    print("\n2. Testing forward pass...")
    features = torch.randn(1, 200)
    harmony_features = encoder.extract_semantic_features(features)
    print(f"   ✅ Extracted harmony features: {harmony_features.shape}")

    # Test parameter extraction
    print("\n3. Testing parameter extraction...")
    params = encoder.extract_harmony_parameters(harmony_features)
    print(f"   ✅ Harmony parameters extracted:")
    print(f"      Major chord frequency: {params.major_chord_frequency:.3f}")
    print(f"      Voicing spread: {params.voicing_spread_preference:.3f}")
    print(f"      ii-V-I frequency: {params.ii_V_I_frequency:.3f}")

    print("\n" + "="*80)
    print("✅ Harmony encoder ready!")
    print("="*80)


if __name__ == "__main__":
    main()
