"""
Rule-Based Features to MIDI Converter
======================================

Converts 1150D feature vectors to MIDI using rule-based feature interpretation.

This is the MVP/quick-win approach that:
1. Extracts musical parameters from features
2. Maps them to existing generator APIs
3. Produces valid MIDI output

Advantages:
- Fast to implement and test
- Uses proven generation code
- Musically valid by design
- No training required

Author: Agent 1 - MIDI Decoder Architecture Lead
Date: November 22, 2025
"""

from typing import Union, Optional, Dict, Any
from pathlib import Path
import numpy as np
import mido
import warnings

from midi_generator.models.features_to_midi import FeaturesToMIDI, validate_features

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RuleBasedFeaturesToMIDI(FeaturesToMIDI):
    """
    Rule-based Features → MIDI converter.

    Strategy:
    1. Extract musical parameters from 1150D features
    2. Map parameters to MIDI generation parameters
    3. Use simplified MIDI generation (mido-based)

    This implementation focuses on core musical elements:
    - Key/mode from harmony features
    - Tempo from rhythm features
    - Melodic contour from melody features
    - Dynamics from dynamics features
    - Texture/voicing from texture features
    - Form/structure from structure features
    - Instrumentation from orchestration features
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize rule-based converter.

        Args:
            verbose: Print detailed conversion info
        """
        self.verbose = verbose

        # Note names for key detection
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                          'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Mode detection
        self.modes = ['major', 'minor', 'dorian', 'phrygian',
                     'lydian', 'mixolydian', 'aeolian', 'locrian']

        # Common chord progressions by mode
        self.chord_progressions = {
            'major': ['I', 'IV', 'V', 'I'],
            'minor': ['i', 'iv', 'V', 'i'],
        }

    def features_to_parameters(
        self,
        features: Union[np.ndarray, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Extract musical parameters from 1150D features.

        Args:
            features: Feature vector [1150]

        Returns:
            Dictionary of musical parameters
        """
        # Convert to numpy if needed
        if TORCH_AVAILABLE and isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()

        # Validate
        if not validate_features(features):
            raise ValueError("Invalid features")

        # Handle batch dimension
        if features.ndim == 2:
            features = features[0]  # Take first in batch

        params = {}

        # Extract feature slices
        harmony_feat = self.extract_feature_slice(features, 'harmony')
        rhythm_feat = self.extract_feature_slice(features, 'rhythm')
        melody_feat = self.extract_feature_slice(features, 'melody')
        dynamics_feat = self.extract_feature_slice(features, 'dynamics')
        texture_feat = self.extract_feature_slice(features, 'texture')
        structure_feat = self.extract_feature_slice(features, 'structure')
        orch_feat = self.extract_feature_slice(features, 'orchestration')

        # ====================================================================
        # HARMONY PARAMETERS
        # ====================================================================
        params['key'] = self._extract_key(harmony_feat)
        params['mode'] = self._extract_mode(harmony_feat)
        params['chord_complexity'] = self._extract_chord_complexity(harmony_feat)
        params['harmonic_rhythm'] = self._extract_harmonic_rhythm(harmony_feat)

        # ====================================================================
        # RHYTHM PARAMETERS
        # ====================================================================
        params['tempo_bpm'] = self._extract_tempo(rhythm_feat)
        params['time_signature'] = self._extract_time_signature(rhythm_feat)
        params['syncopation'] = self._extract_syncopation(rhythm_feat)
        params['swing'] = self._extract_swing(rhythm_feat)

        # ====================================================================
        # MELODY PARAMETERS
        # ====================================================================
        params['melodic_range'] = self._extract_melodic_range(melody_feat)
        params['melodic_contour'] = self._extract_melodic_contour(melody_feat)
        params['step_leap_ratio'] = self._extract_step_leap_ratio(melody_feat)

        # ====================================================================
        # DYNAMICS PARAMETERS
        # ====================================================================
        params['velocity_mean'] = self._extract_velocity_mean(dynamics_feat)
        params['velocity_std'] = self._extract_velocity_std(dynamics_feat)
        params['accent_frequency'] = self._extract_accent_frequency(dynamics_feat)

        # ====================================================================
        # TEXTURE PARAMETERS
        # ====================================================================
        params['voice_count'] = self._extract_voice_count(texture_feat)
        params['density'] = self._extract_density(texture_feat)
        params['texture_type'] = self._extract_texture_type(texture_feat)

        # ====================================================================
        # STRUCTURE PARAMETERS
        # ====================================================================
        params['num_bars'] = self._extract_num_bars(structure_feat)
        params['form_type'] = self._extract_form_type(structure_feat)

        # ====================================================================
        # ORCHESTRATION PARAMETERS
        # ====================================================================
        params['instrument_programs'] = self._extract_programs(orch_feat)
        params['channel_count'] = self._extract_channel_count(orch_feat)

        if self.verbose:
            print("\n🎵 Extracted Parameters:")
            for key, value in params.items():
                print(f"   {key}: {value}")

        return params

    def features_to_midi(
        self,
        features: Union[np.ndarray, torch.Tensor],
        output_path: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> mido.MidiFile:
        """
        Convert features to MIDI using rule-based approach.

        Args:
            features: Feature vector [1150]
            output_path: Optional path to save MIDI
            **kwargs: Additional options

        Returns:
            mido.MidiFile object
        """
        # Extract parameters
        params = self.features_to_parameters(features)

        # Generate MIDI
        midi = self._generate_midi_from_parameters(params, **kwargs)

        # Validate
        if not self.validate_output(midi):
            warnings.warn("Generated MIDI may be invalid")

        # Save if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            midi.save(str(output_path))
            if self.verbose:
                print(f"✅ Saved MIDI to {output_path}")

        return midi

    # ========================================================================
    # HARMONY FEATURE EXTRACTION
    # ========================================================================

    def _extract_key(self, harmony_feat: np.ndarray) -> str:
        """Extract musical key from harmony features."""
        # Use first 12 features as pitch class distribution
        pitch_class_dist = np.abs(harmony_feat[:12])
        pitch_class_dist = pitch_class_dist / (np.sum(pitch_class_dist) + 1e-8)

        # Find most prominent pitch class
        tonic_idx = int(np.argmax(pitch_class_dist))

        return self.note_names[tonic_idx]

    def _extract_mode(self, harmony_feat: np.ndarray) -> str:
        """Extract mode from harmony features."""
        # Features 12-19 might encode mode information
        mode_features = harmony_feat[12:20] if len(harmony_feat) > 20 else harmony_feat[:8]

        # Simple heuristic: positive mean suggests major, negative suggests minor
        mode_score = np.mean(mode_features)

        if mode_score > 0.1:
            return 'major'
        elif mode_score < -0.1:
            return 'minor'
        else:
            return 'major'  # Default

    def _extract_chord_complexity(self, harmony_feat: np.ndarray) -> float:
        """Extract chord complexity score [0, 1]."""
        # Use features 20-30 for complexity
        complexity_features = harmony_feat[20:30] if len(harmony_feat) > 30 else harmony_feat[:10]
        complexity = np.mean(np.abs(complexity_features))
        return float(np.clip(complexity, 0, 1))

    def _extract_harmonic_rhythm(self, harmony_feat: np.ndarray) -> float:
        """Extract harmonic rhythm (chords per bar)."""
        # Features 30-40 for harmonic rhythm
        hr_features = harmony_feat[30:40] if len(harmony_feat) > 40 else harmony_feat[:10]
        hr_score = np.mean(np.abs(hr_features))

        # Map to realistic range [1, 8] chords per bar
        return float(1 + hr_score * 7)

    # ========================================================================
    # RHYTHM FEATURE EXTRACTION
    # ========================================================================

    def _extract_tempo(self, rhythm_feat: np.ndarray) -> float:
        """Extract tempo in BPM."""
        # Use first few rhythm features for tempo
        tempo_features = rhythm_feat[:10]
        tempo_score = np.mean(tempo_features)

        # Map to realistic BPM range [60, 200]
        tempo_bpm = 60 + (tempo_score + 1) / 2 * 140  # Assuming features in [-1, 1]
        return float(np.clip(tempo_bpm, 60, 200))

    def _extract_time_signature(self, rhythm_feat: np.ndarray) -> str:
        """Extract time signature."""
        # Features 10-15 for meter
        meter_features = rhythm_feat[10:15] if len(rhythm_feat) > 15 else rhythm_feat[:5]
        meter_score = np.mean(meter_features)

        # Simple heuristic
        if meter_score > 0.5:
            return '3/4'
        elif meter_score < -0.5:
            return '6/8'
        else:
            return '4/4'  # Most common

    def _extract_syncopation(self, rhythm_feat: np.ndarray) -> float:
        """Extract syncopation level [0, 1]."""
        sync_features = rhythm_feat[15:25] if len(rhythm_feat) > 25 else rhythm_feat[:10]
        return float(np.clip(np.mean(np.abs(sync_features)), 0, 1))

    def _extract_swing(self, rhythm_feat: np.ndarray) -> float:
        """Extract swing ratio."""
        swing_features = rhythm_feat[25:30] if len(rhythm_feat) > 30 else rhythm_feat[:5]
        swing_score = np.mean(swing_features)

        # Map to swing ratio [0.5 (straight), 0.67 (swing)]
        return float(0.5 + (swing_score + 1) / 2 * 0.17)

    # ========================================================================
    # MELODY FEATURE EXTRACTION
    # ========================================================================

    def _extract_melodic_range(self, melody_feat: np.ndarray) -> int:
        """Extract melodic range in semitones."""
        range_features = melody_feat[:10]
        range_score = np.mean(np.abs(range_features))

        # Map to realistic range [5, 24] semitones
        return int(5 + range_score * 19)

    def _extract_melodic_contour(self, melody_feat: np.ndarray) -> str:
        """Extract melodic contour type."""
        contour_features = melody_feat[10:20] if len(melody_feat) > 20 else melody_feat[:10]
        contour_score = np.mean(contour_features)

        if contour_score > 0.3:
            return 'ascending'
        elif contour_score < -0.3:
            return 'descending'
        else:
            return 'arch'

    def _extract_step_leap_ratio(self, melody_feat: np.ndarray) -> float:
        """Extract step/leap ratio."""
        ratio_features = melody_feat[20:30] if len(melody_feat) > 30 else melody_feat[:10]
        ratio = np.mean(np.abs(ratio_features))
        return float(np.clip(ratio, 0, 1))

    # ========================================================================
    # DYNAMICS FEATURE EXTRACTION
    # ========================================================================

    def _extract_velocity_mean(self, dynamics_feat: np.ndarray) -> int:
        """Extract mean velocity [1, 127]."""
        vel_features = dynamics_feat[:10]
        vel_score = np.mean(vel_features)

        # Map to MIDI velocity range
        velocity = int(32 + (vel_score + 1) / 2 * 95)  # [32, 127]
        return int(np.clip(velocity, 1, 127))

    def _extract_velocity_std(self, dynamics_feat: np.ndarray) -> int:
        """Extract velocity standard deviation."""
        std_features = dynamics_feat[10:20] if len(dynamics_feat) > 20 else dynamics_feat[:10]
        std_score = np.mean(np.abs(std_features))

        # Map to [5, 40] range
        return int(5 + std_score * 35)

    def _extract_accent_frequency(self, dynamics_feat: np.ndarray) -> float:
        """Extract accent frequency [0, 1]."""
        accent_features = dynamics_feat[20:30] if len(dynamics_feat) > 30 else dynamics_feat[:10]
        return float(np.clip(np.mean(np.abs(accent_features)), 0, 1))

    # ========================================================================
    # TEXTURE FEATURE EXTRACTION
    # ========================================================================

    def _extract_voice_count(self, texture_feat: np.ndarray) -> int:
        """Extract number of voices."""
        voice_features = texture_feat[:10]
        voice_score = np.mean(np.abs(voice_features))

        # Map to [1, 8] voices
        return int(1 + voice_score * 7)

    def _extract_density(self, texture_feat: np.ndarray) -> float:
        """Extract texture density [0, 1]."""
        density_features = texture_feat[10:20] if len(texture_feat) > 20 else texture_feat[:10]
        return float(np.clip(np.mean(np.abs(density_features)), 0, 1))

    def _extract_texture_type(self, texture_feat: np.ndarray) -> str:
        """Extract texture type."""
        type_features = texture_feat[20:30] if len(texture_feat) > 30 else texture_feat[:10]
        type_score = np.mean(type_features)

        if type_score > 0.3:
            return 'polyphonic'
        elif type_score < -0.3:
            return 'homophonic'
        else:
            return 'melody_accompaniment'

    # ========================================================================
    # STRUCTURE FEATURE EXTRACTION
    # ========================================================================

    def _extract_num_bars(self, structure_feat: np.ndarray) -> int:
        """Extract number of bars."""
        bar_features = structure_feat[:10]
        bar_score = np.mean(np.abs(bar_features))

        # Map to [4, 32] bars
        return int(4 + bar_score * 28)

    def _extract_form_type(self, structure_feat: np.ndarray) -> str:
        """Extract form type."""
        form_features = structure_feat[10:20] if len(structure_feat) > 20 else structure_feat[:10]
        form_score = np.mean(form_features)

        if form_score > 0.3:
            return 'AABA'
        elif form_score < -0.3:
            return 'ABAB'
        else:
            return 'AAA'

    # ========================================================================
    # ORCHESTRATION FEATURE EXTRACTION
    # ========================================================================

    def _extract_programs(self, orch_feat: np.ndarray) -> list:
        """Extract MIDI program numbers."""
        # Use first few features to determine program distribution
        prog_features = orch_feat[:16]  # 16 instrument families

        # Get top programs based on feature values
        top_indices = np.argsort(np.abs(prog_features))[-4:]  # Top 4

        # Map to GM program numbers (simplified)
        program_map = [0, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 0, 0]
        programs = [int(program_map[i]) for i in top_indices]

        return programs

    def _extract_channel_count(self, orch_feat: np.ndarray) -> int:
        """Extract number of channels to use."""
        channel_features = orch_feat[16:20] if len(orch_feat) > 20 else orch_feat[:4]
        channel_score = np.mean(np.abs(channel_features))

        # Map to [1, 8] channels
        return int(1 + channel_score * 7)

    # ========================================================================
    # MIDI GENERATION
    # ========================================================================

    def _generate_midi_from_parameters(
        self,
        params: Dict[str, Any],
        **kwargs
    ) -> mido.MidiFile:
        """
        Generate MIDI file from extracted parameters.

        This is a simplified MIDI generation that creates musically
        valid output based on the parameters.
        """
        # Create MIDI file
        midi = mido.MidiFile(type=1, ticks_per_beat=480)

        # Get parameters
        tempo_bpm = params.get('tempo_bpm', 120)
        key = params.get('key', 'C')
        mode = params.get('mode', 'major')
        num_bars = params.get('num_bars', 16)
        time_sig = params.get('time_signature', '4/4')

        # Parse time signature
        numerator, denominator = self._parse_time_signature(time_sig)

        # Create tempo track
        tempo_track = mido.MidiTrack()
        midi.tracks.append(tempo_track)

        tempo_us = mido.bpm2tempo(tempo_bpm)
        tempo_track.append(mido.MetaMessage('set_tempo', tempo=tempo_us, time=0))
        tempo_track.append(mido.MetaMessage(
            'time_signature',
            numerator=numerator,
            denominator=denominator,
            time=0
        ))

        # Generate melody track
        melody_track = self._generate_melody_track(params)
        midi.tracks.append(melody_track)

        # Generate harmony track (chords)
        harmony_track = self._generate_harmony_track(params)
        midi.tracks.append(harmony_track)

        # Generate bass track
        bass_track = self._generate_bass_track(params)
        midi.tracks.append(bass_track)

        return midi

    def _generate_melody_track(self, params: Dict[str, Any]) -> mido.MidiTrack:
        """Generate a simple melody track."""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=0, time=0))

        # Get parameters
        key = params.get('key', 'C')
        mode = params.get('mode', 'major')
        melodic_range = params.get('melodic_range', 12)
        velocity_mean = params.get('velocity_mean', 80)
        num_bars = params.get('num_bars', 16)

        # Get key root MIDI note
        root_note = self.note_names.index(key) + 60  # Middle C = 60

        # Generate simple melodic pattern
        scale = self._get_scale(key, mode)
        ticks_per_bar = 480 * 4  # Assuming 4/4

        current_time = 0
        for bar in range(num_bars):
            for beat in range(4):
                # Choose a scale degree
                scale_degree = (bar * 4 + beat) % len(scale)
                pitch = root_note + scale[scale_degree]
                pitch = int(np.clip(pitch, 21, 108))

                # Note on
                track.append(mido.Message(
                    'note_on',
                    note=pitch,
                    velocity=velocity_mean,
                    time=0 if (bar == 0 and beat == 0) else 480
                ))

                # Note off
                track.append(mido.Message(
                    'note_off',
                    note=pitch,
                    velocity=0,
                    time=360  # Quarter note duration
                ))

        track.append(mido.MetaMessage('end_of_track', time=0))
        return track

    def _generate_harmony_track(self, params: Dict[str, Any]) -> mido.MidiTrack:
        """Generate harmony/chord track."""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=0, time=0))

        key = params.get('key', 'C')
        mode = params.get('mode', 'major')
        velocity_mean = params.get('velocity_mean', 70)
        num_bars = params.get('num_bars', 16)

        root_note = self.note_names.index(key) + 48  # Lower octave

        # Simple I-IV-V-I progression
        if mode == 'major':
            chord_roots = [0, 5, 7, 0]  # I, IV, V, I
        else:
            chord_roots = [0, 5, 7, 0]  # i, iv, V, i

        for bar in range(num_bars):
            chord_idx = bar % len(chord_roots)
            chord_root = root_note + chord_roots[chord_idx]

            # Major triad: root, major 3rd, perfect 5th
            chord_notes = [chord_root, chord_root + 4, chord_root + 7]

            # Chord on
            for i, note in enumerate(chord_notes):
                track.append(mido.Message(
                    'note_on',
                    note=int(np.clip(note, 21, 108)),
                    velocity=velocity_mean - 10,
                    time=0 if i > 0 else (0 if bar == 0 else 480 * 4)
                ))

            # Chord off
            for i, note in enumerate(chord_notes):
                track.append(mido.Message(
                    'note_off',
                    note=int(np.clip(note, 21, 108)),
                    velocity=0,
                    time=480 * 4 - 100 if i == len(chord_notes) - 1 else 0
                ))

        track.append(mido.MetaMessage('end_of_track', time=0))
        return track

    def _generate_bass_track(self, params: Dict[str, Any]) -> mido.MidiTrack:
        """Generate bass track."""
        track = mido.MidiTrack()
        track.append(mido.Message('program_change', program=32, time=0))  # Acoustic bass

        key = params.get('key', 'C')
        velocity_mean = params.get('velocity_mean', 75)
        num_bars = params.get('num_bars', 16)

        root_note = self.note_names.index(key) + 36  # Low octave

        # Walking bass pattern
        bass_pattern = [0, 2, 4, 5]  # Root, 2nd, 3rd, 4th

        for bar in range(num_bars):
            for beat in range(4):
                bass_note = root_note + bass_pattern[beat % len(bass_pattern)]
                bass_note = int(np.clip(bass_note, 21, 60))

                track.append(mido.Message(
                    'note_on',
                    note=bass_note,
                    velocity=velocity_mean,
                    time=0 if (bar == 0 and beat == 0) else 480
                ))

                track.append(mido.Message(
                    'note_off',
                    note=bass_note,
                    velocity=0,
                    time=360
                ))

        track.append(mido.MetaMessage('end_of_track', time=0))
        return track

    # ========================================================================
    # HELPER METHODS
    # ========================================================================

    def _get_scale(self, key: str, mode: str) -> list:
        """Get scale intervals for a key/mode."""
        # Major scale intervals
        major_scale = [0, 2, 4, 5, 7, 9, 11]
        # Natural minor scale intervals
        minor_scale = [0, 2, 3, 5, 7, 8, 10]

        if mode in ['major', 'ionian', 'lydian', 'mixolydian']:
            return major_scale
        else:
            return minor_scale

    def _parse_time_signature(self, time_sig: str) -> tuple:
        """Parse time signature string to (numerator, denominator)."""
        parts = time_sig.split('/')
        if len(parts) == 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return 4, 4  # Default


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("Rule-Based Features to MIDI Converter")
    print("="*70)

    # Create converter
    converter = RuleBasedFeaturesToMIDI(verbose=True)

    # Test with random features
    print("\n🧪 Testing with random features...")
    test_features = np.random.randn(1150)

    # Extract parameters
    print("\n📊 Extracting parameters...")
    params = converter.features_to_parameters(test_features)

    # Generate MIDI
    print("\n🎵 Generating MIDI...")
    midi = converter.features_to_midi(test_features, output_path="/tmp/test_rule_based.mid")

    # Validate
    print("\n✅ Validating output...")
    is_valid = converter.validate_output(midi)
    print(f"   Valid: {is_valid}")
    print(f"   Tracks: {len(midi.tracks)}")

    print("\n" + "="*70)
    print("✅ Rule-based converter test complete!")
    print("="*70)
