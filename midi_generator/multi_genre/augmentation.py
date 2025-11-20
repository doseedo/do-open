"""
MIDI Augmentation Module
Genre-specific data augmentation for MIDI files

Author: Agent 07
Date: November 20, 2025
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
import copy
import logging

logger = logging.getLogger(__name__)


# Genre-specific augmentation configurations
GENRE_AUGMENTATION_CONFIGS = {
    'jazz': {
        'pitch_transposition': {
            'range': (-5, 7),
            'probability': 0.8
        },
        'tempo_scaling': {
            'range': (0.85, 1.15),
            'probability': 0.7
        },
        'velocity_perturbation': {
            'variance': 10,
            'probability': 0.9
        },
        'timing_jitter': {
            'jitter_ms': 15,
            'probability': 0.8
        },
        'harmonic_substitution': {
            'probability': 0.3
        }
    },
    'classical': {
        'pitch_transposition': {
            'range': (-3, 3),
            'probability': 0.6
        },
        'tempo_scaling': {
            'range': (0.95, 1.05),
            'probability': 0.5
        },
        'velocity_perturbation': {
            'variance': 3,
            'probability': 0.6
        },
        'timing_jitter': {
            'jitter_ms': 5,
            'probability': 0.3
        },
        'harmonic_substitution': {
            'probability': 0.0  # Don't alter classical harmony
        }
    },
    'rock': {
        'pitch_transposition': {
            'range': (-3, 5),
            'probability': 0.7
        },
        'tempo_scaling': {
            'range': (0.9, 1.1),
            'probability': 0.7
        },
        'velocity_perturbation': {
            'variance': 8,
            'probability': 0.8
        },
        'timing_jitter': {
            'jitter_ms': 12,
            'probability': 0.7
        }
    },
    'electronic': {
        'pitch_transposition': {
            'range': (-7, 7),
            'probability': 0.9
        },
        'tempo_scaling': {
            'range': (0.8, 1.2),
            'probability': 0.8
        },
        'velocity_perturbation': {
            'variance': 5,
            'probability': 0.6
        },
        'timing_jitter': {
            'jitter_ms': 2,  # Minimal - quantized
            'probability': 0.3
        }
    },
    'pop': {
        'pitch_transposition': {
            'range': (-4, 4),
            'probability': 0.8
        },
        'tempo_scaling': {
            'range': (0.9, 1.1),
            'probability': 0.7
        },
        'velocity_perturbation': {
            'variance': 6,
            'probability': 0.7
        },
        'timing_jitter': {
            'jitter_ms': 10,
            'probability': 0.6
        }
    }
}


class MIDIAugmentation(ABC):
    """
    Base class for MIDI augmentations.

    All augmentation classes should inherit from this and implement
    the augment() method.
    """

    def __init__(
        self,
        probability: float = 1.0,
        preserve_parameters: Optional[List[str]] = None
    ):
        """
        Initialize augmentation.

        Args:
            probability: Probability of applying augmentation (0.0-1.0)
            preserve_parameters: List of parameter names that must be preserved
        """
        self.probability = probability
        self.preserve_parameters = preserve_parameters or []

    @abstractmethod
    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply augmentation to MIDI data.

        Args:
            midi_data: Dictionary containing MIDI information with keys:
                - 'notes': List of note dicts with 'pitch', 'velocity', 'start', 'end'
                - 'tempo_bpm': Tempo in BPM
                - 'time_signature': Tuple of (numerator, denominator)
                - 'key': Musical key
                - Other metadata

        Returns:
            Augmented MIDI data (new dictionary, original unchanged)
        """
        raise NotImplementedError

    def __call__(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentation with probability."""
        if np.random.random() < self.probability:
            return self.augment(midi_data)
        else:
            return copy.deepcopy(midi_data)

    def validate(
        self,
        original: Dict[str, Any],
        augmented: Dict[str, Any]
    ) -> bool:
        """
        Validate augmentation quality.

        Args:
            original: Original MIDI data
            augmented: Augmented MIDI data

        Returns:
            True if validation passes
        """
        # Basic validation: check structure
        assert 'notes' in augmented, "Missing 'notes' in augmented data"
        assert len(augmented['notes']) > 0, "No notes in augmented data"

        # Check all pitches valid
        for note in augmented['notes']:
            assert 0 <= note['pitch'] <= 127, f"Invalid pitch: {note['pitch']}"
            assert 1 <= note['velocity'] <= 127, f"Invalid velocity: {note['velocity']}"
            assert note['start'] >= 0, f"Negative start time: {note['start']}"
            assert note['end'] > note['start'], f"Invalid duration: {note['end']} <= {note['start']}"

        return True


class PitchTransposition(MIDIAugmentation):
    """
    Transpose MIDI by N semitones.

    Preserves: Intervals, contours, rhythm
    Changes: Absolute pitch, key
    """

    def __init__(
        self,
        semitone_range: Tuple[int, int] = (-3, 3),
        probability: float = 0.8
    ):
        """
        Initialize pitch transposition.

        Args:
            semitone_range: (min, max) semitones to transpose
            probability: Probability of applying
        """
        super().__init__(probability)
        self.semitone_range = semitone_range

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transpose all notes by random semitones."""
        augmented = copy.deepcopy(midi_data)

        # Choose random transposition
        semitones = np.random.randint(
            self.semitone_range[0],
            self.semitone_range[1] + 1
        )

        # Skip if transposition is 0
        if semitones == 0:
            return augmented

        logger.debug(f"Transposing by {semitones} semitones")

        # Transpose all notes
        for note in augmented['notes']:
            new_pitch = note['pitch'] + semitones
            note['pitch'] = np.clip(new_pitch, 0, 127)

        # Update key if present
        if 'key' in augmented:
            augmented['key'] = self._transpose_key(augmented['key'], semitones)

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['transposition'] = semitones

        return augmented

    def _transpose_key(self, key: str, semitones: int) -> str:
        """Transpose musical key by semitones."""
        # Simplified key transposition
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        # Extract tonic and mode
        if len(key) > 1 and key[-1] in ['m', 'M']:
            tonic = key[:-1]
            mode = key[-1]
        else:
            tonic = key
            mode = ''

        # Find current index
        try:
            idx = keys.index(tonic)
        except ValueError:
            logger.warning(f"Unknown key: {key}, not transposing")
            return key

        # Transpose
        new_idx = (idx + semitones) % 12
        new_key = keys[new_idx] + mode

        return new_key


class TempoScaling(MIDIAugmentation):
    """
    Scale tempo while preserving pitch.

    Preserves: Pitch, relative timing
    Changes: Absolute tempo, duration
    """

    def __init__(
        self,
        tempo_range: Tuple[float, float] = (0.9, 1.1),
        probability: float = 0.7
    ):
        """
        Initialize tempo scaling.

        Args:
            tempo_range: (min_factor, max_factor) for tempo scaling
            probability: Probability of applying
        """
        super().__init__(probability)
        self.tempo_range = tempo_range

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Scale tempo by random factor."""
        augmented = copy.deepcopy(midi_data)

        # Choose random scaling factor
        factor = np.random.uniform(self.tempo_range[0], self.tempo_range[1])

        logger.debug(f"Scaling tempo by {factor:.2f}x")

        # Scale all timings
        for note in augmented['notes']:
            note['start'] *= factor
            note['end'] *= factor

        # Update tempo if present
        if 'tempo_bpm' in augmented:
            augmented['tempo_bpm'] /= factor  # Inverse: faster factor = higher tempo

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['tempo_scaling'] = factor

        return augmented


class VelocityPerturbation(MIDIAugmentation):
    """
    Add controlled randomness to velocities.

    Preserves: Articulation, general dynamics
    Changes: Exact velocities
    """

    def __init__(
        self,
        variance: float = 5.0,
        probability: float = 0.8
    ):
        """
        Initialize velocity perturbation.

        Args:
            variance: Standard deviation of Gaussian noise
            probability: Probability of applying
        """
        super().__init__(probability)
        self.variance = variance

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add Gaussian noise to velocities."""
        augmented = copy.deepcopy(midi_data)

        logger.debug(f"Perturbing velocities with variance={self.variance}")

        for note in augmented['notes']:
            noise = np.random.normal(0, self.variance)
            new_velocity = note['velocity'] + int(noise)
            note['velocity'] = np.clip(new_velocity, 1, 127)

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['velocity_perturbation'] = self.variance

        return augmented


class TimingJitter(MIDIAugmentation):
    """
    Add humanization through timing variation.

    Preserves: Rhythm structure, swing
    Changes: Exact timing
    """

    def __init__(
        self,
        jitter_ms: float = 10.0,
        probability: float = 0.7
    ):
        """
        Initialize timing jitter.

        Args:
            jitter_ms: Standard deviation of timing jitter in milliseconds
            probability: Probability of applying
        """
        super().__init__(probability)
        self.jitter_ms = jitter_ms

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Add random timing offsets."""
        augmented = copy.deepcopy(midi_data)

        logger.debug(f"Adding timing jitter: ±{self.jitter_ms}ms")

        # Convert ms to seconds
        jitter_sec = self.jitter_ms / 1000.0

        for note in augmented['notes']:
            offset = np.random.normal(0, jitter_sec)
            duration = note['end'] - note['start']

            note['start'] = max(0, note['start'] + offset)
            note['end'] = note['start'] + duration

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['timing_jitter'] = self.jitter_ms

        return augmented


class HarmonicSubstitution(MIDIAugmentation):
    """
    Apply jazz-style chord substitutions.

    Preserves: Harmonic function
    Changes: Chord voicings

    Note: This is a placeholder implementation.
    Full harmonic substitution requires chord detection and music theory.
    """

    def __init__(
        self,
        substitution_prob: float = 0.2,
        probability: float = 0.3
    ):
        """
        Initialize harmonic substitution.

        Args:
            substitution_prob: Probability of substituting each chord
            probability: Probability of applying augmentation at all
        """
        super().__init__(probability)
        self.substitution_prob = substitution_prob

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply chord substitutions (placeholder)."""
        # TODO: Implement full chord detection and substitution
        # For now, return copy
        logger.debug("Harmonic substitution (placeholder)")
        augmented = copy.deepcopy(midi_data)

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['harmonic_substitution'] = True

        return augmented


class VoicePermutation(MIDIAugmentation):
    """
    Swap instruments while preserving parts.

    Preserves: Notes, timing, structure
    Changes: Instrumentation
    """

    def __init__(self, probability: float = 0.5):
        """
        Initialize voice permutation.

        Args:
            probability: Probability of applying
        """
        super().__init__(probability)

    def augment(self, midi_data: Dict[str, Any]) -> Dict[str, Any]:
        """Permute voice assignments (placeholder)."""
        # TODO: Implement track/instrument permutation
        logger.debug("Voice permutation (placeholder)")
        augmented = copy.deepcopy(midi_data)

        # Add metadata
        augmented['augmentation'] = augmented.get('augmentation', {})
        augmented['augmentation']['voice_permutation'] = True

        return augmented


class GenreAugmentationPipeline:
    """
    Apply genre-appropriate augmentation pipeline.

    Example:
        >>> pipeline = GenreAugmentationPipeline('jazz')
        >>> augmented_files = pipeline.augment(
        ...     midi_data,
        ...     num_variations=4
        ... )
    """

    def __init__(
        self,
        genre: str,
        config: Optional[Dict] = None
    ):
        """
        Initialize genre-specific augmentation pipeline.

        Args:
            genre: Genre name ('jazz', 'classical', 'rock', 'electronic', 'pop')
            config: Custom configuration (uses default if None)
        """
        self.genre = genre.lower()

        # Load configuration
        if config is None:
            if self.genre in GENRE_AUGMENTATION_CONFIGS:
                config = GENRE_AUGMENTATION_CONFIGS[self.genre]
            else:
                logger.warning(f"Unknown genre: {genre}, using default config")
                config = GENRE_AUGMENTATION_CONFIGS['pop']

        self.config = config
        self.pipeline = self._build_pipeline()

        logger.info(f"GenreAugmentationPipeline initialized for {self.genre}")

    def _build_pipeline(self) -> List[MIDIAugmentation]:
        """Build augmentation chain for genre."""
        pipeline = []

        # Pitch transposition
        if 'pitch_transposition' in self.config:
            cfg = self.config['pitch_transposition']
            pipeline.append(PitchTransposition(
                semitone_range=cfg['range'],
                probability=cfg['probability']
            ))

        # Tempo scaling
        if 'tempo_scaling' in self.config:
            cfg = self.config['tempo_scaling']
            pipeline.append(TempoScaling(
                tempo_range=cfg['range'],
                probability=cfg['probability']
            ))

        # Velocity perturbation
        if 'velocity_perturbation' in self.config:
            cfg = self.config['velocity_perturbation']
            pipeline.append(VelocityPerturbation(
                variance=cfg['variance'],
                probability=cfg['probability']
            ))

        # Timing jitter
        if 'timing_jitter' in self.config:
            cfg = self.config['timing_jitter']
            pipeline.append(TimingJitter(
                jitter_ms=cfg['jitter_ms'],
                probability=cfg['probability']
            ))

        # Harmonic substitution (jazz mainly)
        if 'harmonic_substitution' in self.config:
            cfg = self.config['harmonic_substitution']
            if cfg['probability'] > 0:
                pipeline.append(HarmonicSubstitution(
                    probability=cfg['probability']
                ))

        return pipeline

    def augment(
        self,
        midi_data: Dict[str, Any],
        num_variations: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate augmented versions of MIDI data.

        Args:
            midi_data: Original MIDI data
            num_variations: Number of augmented versions to generate

        Returns:
            List of augmented MIDI data dictionaries
        """
        variations = []

        for i in range(num_variations):
            augmented = copy.deepcopy(midi_data)

            # Apply each augmentation in pipeline
            for augmentation in self.pipeline:
                augmented = augmentation(augmented)

            # Add variation metadata
            augmented['variation_id'] = i
            augmented['original_file_id'] = midi_data.get('file_id', 'unknown')

            variations.append(augmented)

        logger.debug(f"Generated {len(variations)} variations for {self.genre}")

        return variations

    def get_augmentation_multiplier(self, train_samples: int, target: int = 500) -> float:
        """
        Calculate how many augmented versions to generate per sample.

        Args:
            train_samples: Number of training samples for this genre
            target: Target number of samples (default 500)

        Returns:
            Augmentation multiplier
        """
        if train_samples == 0:
            return 0
        multiplier = target / train_samples
        return max(1.0, multiplier)


if __name__ == '__main__':
    # Example usage with dummy data
    logging.basicConfig(level=logging.DEBUG)

    # Create dummy MIDI data
    dummy_midi = {
        'file_id': 'test_001',
        'notes': [
            {'pitch': 60, 'velocity': 80, 'start': 0.0, 'end': 0.5},
            {'pitch': 64, 'velocity': 85, 'start': 0.5, 'end': 1.0},
            {'pitch': 67, 'velocity': 90, 'start': 1.0, 'end': 1.5},
        ],
        'tempo_bpm': 120,
        'key': 'C',
        'genre': 'jazz'
    }

    print("Original MIDI:")
    print(f"  Pitches: {[n['pitch'] for n in dummy_midi['notes']]}")
    print(f"  Tempo: {dummy_midi['tempo_bpm']} BPM")
    print(f"  Key: {dummy_midi['key']}")

    # Test jazz pipeline
    pipeline = GenreAugmentationPipeline('jazz')
    variations = pipeline.augment(dummy_midi, num_variations=3)

    print(f"\nGenerated {len(variations)} variations:")
    for i, var in enumerate(variations):
        print(f"\nVariation {i+1}:")
        print(f"  Pitches: {[n['pitch'] for n in var['notes']]}")
        print(f"  Tempo: {var['tempo_bpm']:.1f} BPM")
        print(f"  Key: {var['key']}")
        print(f"  Augmentations: {var.get('augmentation', {})}")

    # Test augmentation multiplier
    print(f"\nAugmentation multipliers:")
    for genre, samples in [('jazz', 105), ('classical', 140), ('rock', 70)]:
        pipeline = GenreAugmentationPipeline(genre)
        multiplier = pipeline.get_augmentation_multiplier(samples, target=500)
        print(f"  {genre}: {multiplier:.2f}x ({samples} → 500)")
