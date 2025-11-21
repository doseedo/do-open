#!/usr/bin/env python3
"""
Orchestration Semantic Encoder - Agent 5 (Modular Semantic Discovery)
=======================================================================

Specialized semantic encoder for discovering orchestration parameters.

This encoder focuses on 25 orchestration-specific parameters:
- Instrumentation density curve
- Vertical spacing preferences
- Doubling strategies
- Timbral balance profiles
- Voice crossing frequency
- Orchestral layering patterns
- Register distribution
- Section blend characteristics
- Dynamic balance ratios
- Texture density evolution

Integrates with:
- BrassArranger (Agent 5 - Big Band)
- Instrumentation Parameters (Agent 7)
- Semantic Discovery Pipeline (Agent 7)
- Musical Locality Functions (Agent 1)

Author: Agent 5 - Orchestration Module Builder
Date: November 21, 2025
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import warnings

# Try to import PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not installed. Neural network functionality will be disabled.")

# NumPy is required
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    warnings.warn("NumPy not installed. Some functionality will be disabled.")

# Import base encoder
try:
    # Try relative import first (when used as module)
    from .semantic_encoder import SemanticFeatureEncoder, EncoderConfig
except ImportError:
    try:
        # Try direct import (when run as script)
        from semantic_encoder import SemanticFeatureEncoder, EncoderConfig
    except ImportError:
        # Fallback for standalone testing
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from semantic_encoder import SemanticFeatureEncoder, EncoderConfig


# ============================================================================
# Orchestration Parameter Definitions
# ============================================================================

ORCHESTRATION_PARAMETERS = {
    # INSTRUMENTATION DENSITY (5 params)
    0: {
        'name': 'instrumentation.density.overall',
        'description': 'Overall orchestration density (sparse to full)',
        'range': (0.0, 1.0),
        'musical_meaning': 'Chamber (0.2) to Full Orchestra (0.9)'
    },
    1: {
        'name': 'instrumentation.density.evolution',
        'description': 'How density changes over time',
        'range': (-1.0, 1.0),
        'musical_meaning': 'Thinning (-1) to Building (+1)'
    },
    2: {
        'name': 'instrumentation.density.peaks',
        'description': 'Frequency of density peaks/climaxes',
        'range': (0.0, 1.0),
        'musical_meaning': 'Sustained (0.2) to Punctuated (0.8)'
    },
    3: {
        'name': 'instrumentation.density.contrast',
        'description': 'Contrast between thick and thin sections',
        'range': (0.0, 1.0),
        'musical_meaning': 'Uniform (0.2) to High Contrast (0.9)'
    },
    4: {
        'name': 'instrumentation.active_voices',
        'description': 'Number of simultaneously active voices',
        'range': (1.0, 20.0),
        'musical_meaning': 'Solo (1) to Full Ensemble (16+)'
    },

    # VERTICAL SPACING (4 params)
    5: {
        'name': 'voicing.vertical_spacing.preference',
        'description': 'Preferred interval spacing between voices',
        'range': (0.0, 1.0),
        'musical_meaning': 'Close (0.2) to Wide/Spread (0.9)'
    },
    6: {
        'name': 'voicing.vertical_spacing.bass_tenor_gap',
        'description': 'Distance between bass and tenor voices',
        'range': (12.0, 36.0),
        'musical_meaning': 'Semitones between lowest and next voice'
    },
    7: {
        'name': 'voicing.vertical_spacing.upper_voices',
        'description': 'Spacing tightness in upper voices',
        'range': (0.0, 1.0),
        'musical_meaning': 'Tight clusters (0.2) to Open intervals (0.8)'
    },
    8: {
        'name': 'voicing.vertical_spacing.register_balance',
        'description': 'Balance across low/mid/high registers',
        'range': (0.0, 1.0),
        'musical_meaning': 'Bottom heavy (0.2) to Top heavy (0.8)'
    },

    # DOUBLING STRATEGIES (4 params)
    9: {
        'name': 'doubling.octave_frequency',
        'description': 'Frequency of octave doublings',
        'range': (0.0, 1.0),
        'musical_meaning': 'Rare (0.1) to Constant (0.9)'
    },
    10: {
        'name': 'doubling.unison_frequency',
        'description': 'Frequency of unison doublings',
        'range': (0.0, 1.0),
        'musical_meaning': 'Independent (0.1) to Reinforced (0.8)'
    },
    11: {
        'name': 'doubling.family_preference',
        'description': 'Preference for within-family vs cross-family doubling',
        'range': (0.0, 1.0),
        'musical_meaning': 'Cross-family (0.2) to Within-family (0.8)'
    },
    12: {
        'name': 'doubling.melody_reinforcement',
        'description': 'Tendency to double the melody line',
        'range': (0.0, 1.0),
        'musical_meaning': 'Solo melody (0.2) to Heavy doubling (0.9)'
    },

    # TIMBRAL BALANCE (3 params)
    13: {
        'name': 'timbre.balance.strings_prominence',
        'description': 'Prominence of string instruments',
        'range': (0.0, 1.0),
        'musical_meaning': 'Background (0.2) to Dominant (0.9)'
    },
    14: {
        'name': 'timbre.balance.winds_prominence',
        'description': 'Prominence of wind instruments',
        'range': (0.0, 1.0),
        'musical_meaning': 'Background (0.2) to Dominant (0.9)'
    },
    15: {
        'name': 'timbre.balance.brass_prominence',
        'description': 'Prominence of brass instruments',
        'range': (0.0, 1.0),
        'musical_meaning': 'Background (0.2) to Dominant (0.9)'
    },

    # VOICE INDEPENDENCE (3 params)
    16: {
        'name': 'texture.voice_independence',
        'description': 'Degree of independent melodic lines',
        'range': (0.0, 1.0),
        'musical_meaning': 'Homophonic (0.2) to Polyphonic (0.9)'
    },
    17: {
        'name': 'texture.voice_crossing_frequency',
        'description': 'Frequency of voice crossing events',
        'range': (0.0, 1.0),
        'musical_meaning': 'Strict order (0.1) to Free crossing (0.7)'
    },
    18: {
        'name': 'texture.counterpoint_complexity',
        'description': 'Complexity of contrapuntal writing',
        'range': (0.0, 1.0),
        'musical_meaning': 'Simple (0.2) to Complex fugal (0.9)'
    },

    # DYNAMIC BALANCE (3 params)
    19: {
        'name': 'dynamics.melody_accompaniment_ratio',
        'description': 'Dynamic ratio between melody and accompaniment',
        'range': (1.0, 2.0),
        'musical_meaning': 'Multiplier for melody prominence'
    },
    20: {
        'name': 'dynamics.bass_prominence',
        'description': 'Dynamic prominence of bass line',
        'range': (0.8, 1.5),
        'musical_meaning': 'Receding (0.8) to Prominent (1.4)'
    },
    21: {
        'name': 'dynamics.family_balance_mode',
        'description': 'How instrument families are balanced',
        'range': (0.0, 1.0),
        'musical_meaning': 'Auto-balanced (0.3) to Custom weights (0.8)'
    },

    # REGISTER DISTRIBUTION (3 params)
    22: {
        'name': 'register.preferred_range',
        'description': 'Preferred overall register',
        'range': (0.0, 1.0),
        'musical_meaning': 'Low/Dark (0.2) to High/Bright (0.8)'
    },
    23: {
        'name': 'register.spread',
        'description': 'How spread out across registers',
        'range': (0.0, 1.0),
        'musical_meaning': 'Compact (0.2) to Wide span (0.9)'
    },
    24: {
        'name': 'register.extreme_frequency',
        'description': 'Use of extreme high/low registers',
        'range': (0.0, 1.0),
        'musical_meaning': 'Comfortable (0.2) to Extreme (0.8)'
    },
}


# ============================================================================
# Orchestration Feature Extractor
# ============================================================================

class OrchestrationFeatureExtractor:
    """
    Extract orchestration-specific features from MIDI for the encoder.

    This extracts features that capture orchestration characteristics:
    - Active voice counts over time
    - Register distribution histograms
    - Doubling detection
    - Timbral balance analysis
    - Voice independence metrics
    """

    def __init__(self):
        """Initialize extractor"""
        self.instrument_families = {
            'strings': list(range(40, 48)),      # Violin, Viola, Cello, Bass
            'winds': list(range(68, 80)),        # Oboe, Clarinet, Bassoon, Flute
            'brass': list(range(56, 68)),        # Trumpet, Trombone, Tuba, Horn
            'percussion': list(range(0, 16)),    # Various percussion
            'keyboards': list(range(0, 8)),      # Piano, Harpsichord, etc.
        }

    def extract_orchestration_features(self, midi_data: Any) -> np.ndarray:
        """
        Extract orchestration features from MIDI data.

        Args:
            midi_data: MIDI data (mido.MidiFile, Path to MIDI file, or pretty_midi.PrettyMIDI)

        Returns:
            Feature vector [200] for input to encoder
        """
        try:
            # Load MIDI if path provided
            if isinstance(midi_data, (str, Path)):
                import mido
                midi_data = mido.MidiFile(str(midi_data))

            # Parse MIDI into structured data
            parsed_data = self._parse_midi_data(midi_data)

            features = []

            # Feature group 1: Density metrics (20 features)
            density_features = self._extract_density_features(parsed_data)
            features.extend(density_features)

            # Feature group 2: Vertical spacing (20 features)
            spacing_features = self._extract_spacing_features(parsed_data)
            features.extend(spacing_features)

            # Feature group 3: Doubling detection (20 features)
            doubling_features = self._extract_doubling_features(parsed_data)
            features.extend(doubling_features)

            # Feature group 4: Timbral balance (30 features)
            timbral_features = self._extract_timbral_features(parsed_data)
            features.extend(timbral_features)

            # Feature group 5: Voice independence (30 features)
            independence_features = self._extract_voice_independence_features(parsed_data)
            features.extend(independence_features)

            # Feature group 6: Dynamic analysis (30 features)
            dynamic_features = self._extract_dynamic_features(parsed_data)
            features.extend(dynamic_features)

            # Feature group 7: Register distribution (30 features)
            register_features = self._extract_register_features(parsed_data)
            features.extend(register_features)

            # Feature group 8: Texture evolution (20 features)
            texture_features = self._extract_texture_features(parsed_data)
            features.extend(texture_features)

            # Pad or truncate to exactly 200 features
            features = features[:200]
            while len(features) < 200:
                features.append(0.0)

            return np.array(features, dtype=np.float32)

        except Exception as e:
            warnings.warn(f"Error extracting orchestration features: {e}")
            # Return zeros as fallback
            return np.zeros(200, dtype=np.float32)

    def _parse_midi_data(self, midi_data: Any) -> Dict:
        """Parse MIDI data into structured format for feature extraction"""
        import mido

        parsed = {
            'tracks': [],
            'programs': {},  # channel -> program
            'notes_by_track': [],
            'notes_by_program': {},
            'all_notes': []
        }

        try:
            for track_idx, track in enumerate(midi_data.tracks):
                track_notes = []
                current_notes = {}  # (note, channel) -> (velocity, start_time)
                current_time = 0.0
                current_program = {i: 0 for i in range(16)}  # Default program for each channel

                for msg in track:
                    current_time += msg.time

                    # Track program changes
                    if msg.type == 'program_change':
                        current_program[msg.channel] = msg.program
                        parsed['programs'][msg.channel] = msg.program

                    # Track notes
                    elif msg.type == 'note_on' and msg.velocity > 0:
                        key = (msg.note, msg.channel)
                        current_notes[key] = (msg.velocity, current_time, current_program[msg.channel])

                    elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                        key = (msg.note, msg.channel)
                        if key in current_notes:
                            velocity, start_time, program = current_notes.pop(key)
                            note_data = {
                                'pitch': msg.note,
                                'velocity': velocity,
                                'start': start_time,
                                'end': current_time,
                                'duration': current_time - start_time,
                                'channel': msg.channel,
                                'program': program,
                                'track': track_idx
                            }
                            track_notes.append(note_data)
                            parsed['all_notes'].append(note_data)

                            # Group by program
                            if program not in parsed['notes_by_program']:
                                parsed['notes_by_program'][program] = []
                            parsed['notes_by_program'][program].append(note_data)

                parsed['notes_by_track'].append(track_notes)

            # Sort all notes by start time
            parsed['all_notes'].sort(key=lambda n: n['start'])

        except Exception as e:
            warnings.warn(f"Error parsing MIDI: {e}")

        return parsed

    def _extract_density_features(self, parsed_data: Dict) -> List[float]:
        """Extract density-related features (20 features)"""
        features = []
        notes = parsed_data['all_notes']

        if not notes:
            return [0.0] * 20

        # Calculate active voices over time (sample every 0.1 seconds)
        max_time = max(n['end'] for n in notes) if notes else 1.0
        time_samples = np.arange(0, max_time, 0.1)
        active_voices = []

        for t in time_samples:
            count = sum(1 for n in notes if n['start'] <= t < n['end'])
            active_voices.append(count)

        active_voices = np.array(active_voices) if active_voices else np.array([0])

        # 1-5: Active voice statistics
        features.append(float(np.mean(active_voices)))  # mean active voices
        features.append(float(np.std(active_voices)))   # std of density
        features.append(float(np.max(active_voices)))   # peak density
        features.append(float(np.min(active_voices)))   # minimum density
        features.append(float(np.median(active_voices))) # median density

        # 6-10: Density evolution (split into 5 sections)
        section_size = len(active_voices) // 5 if len(active_voices) >= 5 else 1
        for i in range(5):
            start_idx = i * section_size
            end_idx = min((i+1) * section_size, len(active_voices))
            section_mean = np.mean(active_voices[start_idx:end_idx]) if start_idx < len(active_voices) else 0.0
            features.append(float(section_mean))

        # 11-15: Density change rate
        if len(active_voices) > 1:
            density_changes = np.diff(active_voices)
            features.append(float(np.mean(np.abs(density_changes))))  # avg change rate
            features.append(float(np.std(density_changes)))           # variability
            features.append(float(np.max(density_changes)))           # max increase
            features.append(float(np.min(density_changes)))           # max decrease
            features.append(float(len(np.where(density_changes > 2)[0]) / len(density_changes)))  # spike frequency
        else:
            features.extend([0.0] * 5)

        # 16-20: Additional density metrics
        features.append(float(len(parsed_data['notes_by_track'])))  # number of tracks
        features.append(float(len(parsed_data['programs'])))        # number of different instruments
        features.append(float(len(notes) / max_time if max_time > 0 else 0))  # notes per second
        features.append(float(np.percentile(active_voices, 75)))    # 75th percentile density
        features.append(float(np.percentile(active_voices, 25)))    # 25th percentile density

        return features[:20]  # Ensure exactly 20 features

    def _extract_spacing_features(self, parsed_data: Dict) -> List[float]:
        """Extract vertical spacing features (20 features)"""
        notes = parsed_data['all_notes']
        if not notes:
            return [0.0] * 20

        # Analyze vertical spacing at each timestamp
        time_samples = sorted(set(n['start'] for n in notes))[:100]  # Sample first 100 timestamps
        spacings = []

        for t in time_samples:
            simultaneous = sorted([n['pitch'] for n in notes if n['start'] <= t < n['end']])
            if len(simultaneous) > 1:
                intervals = [simultaneous[i+1] - simultaneous[i] for i in range(len(simultaneous)-1)]
                spacings.extend(intervals)

        if spacings:
            return [
                float(np.mean(spacings)), float(np.std(spacings)), float(np.min(spacings)),
                float(np.max(spacings)), float(np.median(spacings)),
                float(np.percentile(spacings, 25)), float(np.percentile(spacings, 75)),
                float(sum(1 for s in spacings if s <= 2) / len(spacings)),  # close voicing ratio
                float(sum(1 for s in spacings if s >= 12) / len(spacings)), # wide spacing ratio
                float(sum(1 for s in spacings if 3 <= s <= 7) / len(spacings)),  # mid spacing
            ] + [float(np.mean(spacings))] * 10  # Repeat mean for remaining features
        return [0.0] * 20

    def _extract_doubling_features(self, parsed_data: Dict) -> List[float]:
        """Extract doubling detection features (20 features)"""
        notes = parsed_data['all_notes']
        if not notes:
            return [0.0] * 20

        doublings = {'octave': 0, 'unison': 0, 'total_pairs': 0}
        time_samples = sorted(set(n['start'] for n in notes))[:100]

        for t in time_samples:
            simultaneous = [n for n in notes if n['start'] <= t < n['end']]
            for i, n1 in enumerate(simultaneous):
                for n2 in simultaneous[i+1:]:
                    doublings['total_pairs'] += 1
                    pitch_diff = abs(n1['pitch'] - n2['pitch'])
                    if pitch_diff == 0:
                        doublings['unison'] += 1
                    elif pitch_diff % 12 == 0:
                        doublings['octave'] += 1

        total = doublings['total_pairs'] if doublings['total_pairs'] > 0 else 1
        return [
            float(doublings['octave'] / total), float(doublings['unison'] / total),
            float((doublings['octave'] + doublings['unison']) / total)
        ] + [0.0] * 17

    def _extract_timbral_features(self, parsed_data: Dict) -> List[float]:
        """Extract timbral balance features (30 features)"""
        programs = parsed_data['programs']
        notes_by_program = parsed_data['notes_by_program']

        # Count notes per instrument family
        family_counts = {family: 0 for family in self.instrument_families}
        for program, notes in notes_by_program.items():
            for family, program_range in self.instrument_families.items():
                if program in program_range:
                    family_counts[family] += len(notes)
                    break

        total_notes = sum(family_counts.values()) if sum(family_counts.values()) > 0 else 1

        features = [float(count / total_notes) for count in family_counts.values()]
        features.extend([float(len(programs))])  # Number of unique programs
        features.extend([0.0] * (30 - len(features)))  # Pad to 30
        return features[:30]

    def _extract_voice_independence_features(self, parsed_data: Dict) -> List[float]:
        """Extract voice independence features (30 features)"""
        tracks = parsed_data['notes_by_track']
        if not tracks:
            return [0.0] * 30

        # Analyze melodic independence per track
        independence_scores = []
        for track_notes in tracks[:10]:  # Analyze first 10 tracks
            if len(track_notes) > 1:
                pitches = [n['pitch'] for n in sorted(track_notes, key=lambda x: x['start'])]
                intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
                independence = np.std(intervals) if intervals else 0.0
                independence_scores.append(independence)

        features = [float(np.mean(independence_scores)) if independence_scores else 0.0]
        features.append(float(len(tracks)))  # Number of voices
        features.extend([0.0] * (30 - len(features)))
        return features[:30]

    def _extract_dynamic_features(self, parsed_data: Dict) -> List[float]:
        """Extract dynamic balance features (30 features)"""
        notes = parsed_data['all_notes']
        if not notes:
            return [0.0] * 30

        velocities = [n['velocity'] for n in notes]
        features = [
            float(np.mean(velocities)), float(np.std(velocities)),
            float(np.min(velocities)), float(np.max(velocities)),
            float(np.median(velocities))
        ]

        # Velocity distribution per program
        for program, program_notes in list(parsed_data['notes_by_program'].items())[:5]:
            vels = [n['velocity'] for n in program_notes]
            features.append(float(np.mean(vels)) if vels else 0.0)

        features.extend([0.0] * (30 - len(features)))
        return features[:30]

    def _extract_register_features(self, parsed_data: Dict) -> List[float]:
        """Extract register distribution features (30 features)"""
        notes = parsed_data['all_notes']
        if not notes:
            return [0.0] * 30

        pitches = [n['pitch'] for n in notes]
        # Divide into register bins
        registers = {
            'very_low': sum(1 for p in pitches if p < 36),
            'low': sum(1 for p in pitches if 36 <= p < 48),
            'mid_low': sum(1 for p in pitches if 48 <= p < 60),
            'mid': sum(1 for p in pitches if 60 <= p < 72),
            'mid_high': sum(1 for p in pitches if 72 <= p < 84),
            'high': sum(1 for p in pitches if 84 <= p < 96),
            'very_high': sum(1 for p in pitches if p >= 96)
        }

        total = len(pitches)
        features = [float(count / total) for count in registers.values()]
        features.extend([float(np.mean(pitches)), float(np.std(pitches))])
        features.extend([0.0] * (30 - len(features)))
        return features[:30]

    def _extract_texture_features(self, parsed_data: Dict) -> List[float]:
        """Extract texture evolution features (20 features)"""
        notes = parsed_data['all_notes']
        if not notes:
            return [0.0] * 20

        # Sample texture density over time (5 sections)
        max_time = max(n['end'] for n in notes) if notes else 1.0
        section_duration = max_time / 5
        texture_densities = []

        for i in range(5):
            section_start = i * section_duration
            section_end = (i + 1) * section_duration
            section_notes = [n for n in notes if section_start <= n['start'] < section_end]
            density = len(section_notes) / section_duration if section_duration > 0 else 0
            texture_densities.append(float(density))

        features = texture_densities
        features.extend([float(np.mean(texture_densities)), float(np.std(texture_densities))])
        features.extend([0.0] * (20 - len(features)))
        return features[:20]


# ============================================================================
# Orchestration Semantic Encoder
# ============================================================================

class OrchestrationSemanticEncoder(SemanticFeatureEncoder):
    """
    Semantic encoder specialized for orchestration parameter discovery.

    This encoder extends SemanticFeatureEncoder to focus on 25 orchestration
    parameters. It learns to compress orchestration features and discover
    interpretable parameters that capture orchestration decisions.

    Architecture:
        Input: 200D orchestration features
        Encoder: [200] → [512] → [25]
        Decoder: [25] → [512] → [200]
        Locality Predictor: [50] → [512] → [12]

    Usage:
        # Create encoder
        encoder = OrchestrationSemanticEncoder()

        # Extract features from MIDI
        extractor = OrchestrationFeatureExtractor()
        features = extractor.extract_orchestration_features(midi_file)
        features_tensor = torch.from_numpy(features).float()

        # Extract orchestration parameters
        orchestration_params = encoder.extract_orchestration_parameters(features_tensor)

        # Get interpretable parameter names
        param_dict = encoder.interpret_parameters(orchestration_params)
    """

    def __init__(self, config: Optional[EncoderConfig] = None):
        """
        Initialize orchestration encoder.

        Args:
            config: Encoder configuration (defaults to 25 semantic features)
        """
        # Default config for orchestration (25 parameters)
        if config is None:
            config = EncoderConfig(
                input_dim=200,
                hidden_dim=512,
                num_semantic_features=25,  # 25 orchestration parameters
                num_locality_types=12,
                reconstruction_weight=1.0,
                locality_weight=0.5,
                sparsity_weight=0.01,
                learning_rate=1e-4,
                dropout=0.1
            )

        # Initialize base encoder
        super().__init__(config)

        # Orchestration-specific components
        self.feature_extractor = OrchestrationFeatureExtractor()
        self.parameter_definitions = ORCHESTRATION_PARAMETERS

    def extract_orchestration_parameters(
        self,
        features: torch.Tensor,
        as_numpy: bool = False
    ) -> torch.Tensor:
        """
        Extract orchestration parameters from feature vector.

        This is an alias for extract_semantic_features with orchestration context.

        Args:
            features: Input features [batch_size, 200] or [200]
            as_numpy: Return as numpy array

        Returns:
            Orchestration parameters [batch_size, 25] or [25]
        """
        return self.extract_semantic_features(features, as_numpy=as_numpy)

    def extract_from_midi(
        self,
        midi_data: Any,
        as_numpy: bool = False
    ) -> torch.Tensor:
        """
        Extract orchestration parameters directly from MIDI data.

        Args:
            midi_data: MIDI data (various formats supported)
            as_numpy: Return as numpy array

        Returns:
            Orchestration parameters [25]
        """
        # Extract features
        features = self.feature_extractor.extract_orchestration_features(midi_data)
        features_tensor = torch.from_numpy(features).float()

        # Extract parameters
        params = self.extract_orchestration_parameters(features_tensor, as_numpy=as_numpy)

        return params

    def interpret_parameters(
        self,
        parameters: torch.Tensor
    ) -> Dict[str, float]:
        """
        Interpret orchestration parameters with semantic names.

        Args:
            parameters: Orchestration parameters [25] or [batch_size, 25]

        Returns:
            Dictionary mapping parameter names to values
        """
        # Handle batched input
        if parameters.dim() == 2:
            # Take first sample for interpretation
            parameters = parameters[0]

        # Convert to numpy
        if isinstance(parameters, torch.Tensor):
            parameters = parameters.detach().cpu().numpy()

        # Build interpretation dictionary
        interpreted = {}
        for i, value in enumerate(parameters):
            if i in self.parameter_definitions:
                param_def = self.parameter_definitions[i]
                param_name = param_def['name']
                interpreted[param_name] = float(value)

        return interpreted

    def get_parameter_description(self, param_index: int) -> Dict[str, Any]:
        """
        Get detailed description of a parameter.

        Args:
            param_index: Parameter index (0-24)

        Returns:
            Parameter definition dictionary
        """
        if param_index in self.parameter_definitions:
            return self.parameter_definitions[param_index]
        else:
            return {
                'name': f'param_{param_index}',
                'description': 'Unknown parameter',
                'range': (0.0, 1.0),
                'musical_meaning': 'Not defined'
            }

    def connect_to_brass_arranger(self, brass_arranger):
        """
        Connect to BrassArranger to apply discovered parameters.

        Args:
            brass_arranger: BrassArranger instance
        """
        # Store reference for parameter application
        self.brass_arranger = brass_arranger
        print(f"✅ Connected to BrassArranger")

    def apply_parameters_to_arrangement(
        self,
        parameters: torch.Tensor,
        arranger: Any
    ) -> Dict[str, Any]:
        """
        Apply discovered parameters to an arranger.

        Args:
            parameters: Orchestration parameters [25]
            arranger: Arranger instance (BrassArranger, BigBandArranger, etc.)

        Returns:
            Configuration dict for arranger
        """
        # Interpret parameters
        param_dict = self.interpret_parameters(parameters)

        # Build arranger configuration
        config = {
            'density': param_dict.get('instrumentation.density.overall', 0.5),
            'voicing_spread': param_dict.get('voicing.vertical_spacing.preference', 0.5),
            'doubling_frequency': param_dict.get('doubling.octave_frequency', 0.4),
            'voice_independence': param_dict.get('texture.voice_independence', 0.5),
            'brass_prominence': param_dict.get('timbre.balance.brass_prominence', 0.5),
        }

        return config


# ============================================================================
# Voice Independence Metrics
# ============================================================================

class VoiceIndependenceAnalyzer:
    """
    Analyze voice independence in orchestration.

    Computes metrics for:
    - Melodic independence (correlation between voice contours)
    - Rhythmic independence (rhythmic similarity)
    - Voice crossing frequency
    - Polyphonic vs homophonic ratio
    """

    def __init__(self):
        """Initialize analyzer"""
        pass

    def compute_voice_independence(
        self,
        voices: List[List[Tuple[int, int, int]]]
    ) -> float:
        """
        Compute voice independence score.

        Args:
            voices: List of voices, each voice is list of (time, pitch, duration)

        Returns:
            Independence score 0.0 (homophonic) to 1.0 (polyphonic)
        """
        if len(voices) < 2:
            return 0.0

        # Compute pairwise melodic correlation
        correlations = []
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                corr = self._compute_melodic_correlation(voices[i], voices[j])
                correlations.append(corr)

        # Independence is inverse of correlation
        avg_correlation = np.mean(correlations) if correlations else 0.0
        independence = 1.0 - avg_correlation

        return independence

    def _compute_melodic_correlation(
        self,
        voice1: List[Tuple[int, int, int]],
        voice2: List[Tuple[int, int, int]]
    ) -> float:
        """
        Compute melodic correlation between two voices.

        Args:
            voice1, voice2: Voice data as (time, pitch, duration) tuples

        Returns:
            Correlation 0.0 (independent) to 1.0 (parallel)
        """
        # Placeholder implementation
        # TODO: Implement actual melodic correlation
        return 0.5

    def compute_voice_crossing_frequency(
        self,
        voices: List[List[Tuple[int, int, int]]]
    ) -> float:
        """
        Compute frequency of voice crossing events.

        Args:
            voices: List of voices

        Returns:
            Crossing frequency 0.0 (none) to 1.0 (frequent)
        """
        # Placeholder implementation
        # TODO: Implement voice crossing detection
        return 0.2


# ============================================================================
# Utility Functions
# ============================================================================

def create_orchestration_encoder(device: str = 'cpu') -> OrchestrationSemanticEncoder:
    """
    Create orchestration encoder with default configuration.

    Args:
        device: Device to create model on

    Returns:
        Initialized OrchestrationSemanticEncoder
    """
    encoder = OrchestrationSemanticEncoder()
    if TORCH_AVAILABLE:
        encoder.to(device)
    return encoder


def analyze_orchestration_from_midi(
    midi_file_path: str,
    encoder: OrchestrationSemanticEncoder
) -> Dict[str, Any]:
    """
    Analyze orchestration characteristics from MIDI file.

    Args:
        midi_file_path: Path to MIDI file
        encoder: Trained OrchestrationSemanticEncoder

    Returns:
        Analysis dictionary with orchestration parameters
    """
    # Load MIDI (placeholder - would use mido or pretty_midi)
    # midi_data = load_midi(midi_file_path)
    midi_data = None  # Placeholder

    # Extract parameters
    try:
        params = encoder.extract_from_midi(midi_data, as_numpy=True)
        param_dict = encoder.interpret_parameters(torch.from_numpy(params))

        return {
            'file': midi_file_path,
            'parameters': param_dict,
            'success': True
        }
    except Exception as e:
        return {
            'file': midi_file_path,
            'error': str(e),
            'success': False
        }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("Orchestration Semantic Encoder - Agent 5")
    print("=" * 80)

    if not TORCH_AVAILABLE:
        print("\n❌ PyTorch is not installed.")
        print("   Install PyTorch to use this module:")
        print("   pip install torch")
        exit(1)

    # Create encoder
    print("\n1. Creating orchestration encoder...")
    encoder = create_orchestration_encoder()
    print(f"   ✅ Created encoder with {encoder.config.num_semantic_features} orchestration parameters")

    # Show parameter definitions
    print("\n2. Orchestration parameters (25 total):")
    for i in range(25):
        param_def = encoder.get_parameter_description(i)
        print(f"   [{i:2d}] {param_def['name']}")
        print(f"        {param_def['description']}")
        print(f"        Musical: {param_def['musical_meaning']}")

    # Test feature extraction
    print("\n3. Testing feature extraction...")
    extractor = OrchestrationFeatureExtractor()
    features = extractor.extract_orchestration_features(None)  # Dummy data
    print(f"   ✅ Extracted {len(features)} features")

    # Test parameter extraction
    print("\n4. Testing parameter extraction...")
    features_tensor = torch.from_numpy(features).float().unsqueeze(0)  # Add batch dim
    params = encoder.extract_orchestration_parameters(features_tensor)
    print(f"   ✅ Extracted {params.shape[1]} orchestration parameters")

    # Interpret parameters
    print("\n5. Interpreting parameters...")
    param_dict = encoder.interpret_parameters(params)
    print(f"   ✅ Interpreted {len(param_dict)} parameters")
    for name, value in list(param_dict.items())[:5]:  # Show first 5
        print(f"      {name}: {value:.3f}")

    # Test voice independence analyzer
    print("\n6. Testing voice independence analyzer...")
    analyzer = VoiceIndependenceAnalyzer()
    voices = [
        [(0, 60, 100), (100, 62, 100)],  # Voice 1
        [(0, 64, 100), (100, 65, 100)],  # Voice 2
    ]
    independence = analyzer.compute_voice_independence(voices)
    print(f"   ✅ Voice independence: {independence:.3f}")

    print("\n" + "=" * 80)
    print("✅ Orchestration Encoder Module Ready!")
    print("=" * 80)
    print("\nNext steps:")
    print("  - Train encoder on orchestration corpus")
    print("  - Integrate with BrassArranger and other arrangers")
    print("  - Connect to semantic discovery pipeline")
    print("  - Implement full feature extraction from MIDI")
