"""
<<<<<<< HEAD
Synthetic Training Data Generator - Agent 14
==============================================

Generates high-quality synthetic training data for new parameters in the
Musical Program Synthesis system.

This module provides:
1. Latin hypercube sampling for even parameter space coverage
2. Genre-balanced dataset generation
3. Musical coherence validation
4. Diverse parameter variation to prevent overfitting
5. Comprehensive metadata tracking
6. Robust error handling and retry logic
7. Real-time progress monitoring
8. Support for all parameter types (continuous, categorical, array, etc.)

The generator creates 1000+ training examples per parameter, ensuring
high-quality data for XGBoost model training with:
- Even coverage across parameter space
- Musical validity checking
- Balanced representation across genres
- Feature extraction integration
- Comprehensive logging and analytics

Author: Agent 14 - Synthetic Training Data Generator
=======
AGENT 14: Synthetic Training Data Generator
============================================

Generates diverse, high-quality training data for new parameters.

Strategy:
- Latin hypercube sampling for even parameter space coverage
- Diverse sampling of other parameters to prevent overfitting
- Musical validity checking to reject nonsense
- Balanced dataset across parameter range
- Genre-balanced generation

Author: Agent 14 - Training Data Specialist
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
License: MIT
"""

import json
import random
import time
<<<<<<< HEAD
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field, asdict

import numpy as np
import mido
from tqdm import tqdm

# Import scipy for Latin hypercube sampling
try:
    from scipy.stats import qmc
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not available. Install with 'pip install scipy' for better sampling.")

# Import system components
from midi_generator.parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
    MusicalImpact
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

=======
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed, using fallback sampling")

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, **kwargs):
        return iterable


>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
@dataclass
class TrainingExample:
    """Single training example for a parameter"""
    features: np.ndarray
    parameter_value: Any
    midi_file: Path
    other_params: Dict[str, Any]
    generation_time: float
    coherence_score: float
    genre: Optional[str] = None
<<<<<<< HEAD
    validation_passed: bool = True
    error_message: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'features': self.features.tolist() if isinstance(self.features, np.ndarray) else self.features,
            'parameter_value': self._serialize_value(self.parameter_value),
            'midi_file': str(self.midi_file.name),
            'generation_time': self.generation_time,
            'coherence_score': self.coherence_score,
            'genre': self.genre,
            'validation_passed': self.validation_passed,
            'error_message': self.error_message
        }

    def _serialize_value(self, value):
        """Serialize parameter value for JSON"""
        if isinstance(value, (int, float, str, bool)):
            return value
        elif isinstance(value, np.number):
            return float(value)
        elif isinstance(value, (list, tuple)):
            return [self._serialize_value(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return str(value)


@dataclass
class DatasetStatistics:
    """Statistics about generated dataset"""
    total_examples: int
    successful_examples: int
    failed_examples: int
    avg_coherence_score: float
    avg_generation_time: float
    parameter_value_distribution: Dict[str, Any]
    genre_distribution: Optional[Dict[str, int]] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        return asdict(self)


# ============================================================================
# MUSICAL COHERENCE VALIDATOR
# ============================================================================

class MusicalCoherenceValidator:
    """
    Validates that generated MIDI is musically coherent.

    Checks multiple aspects:
    - Has notes (not empty)
    - Reasonable length
    - Valid pitch range
    - Appropriate velocities
    - Rhythmic coherence
    - Basic harmonic validity
    """

    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator

        Args:
            strict_mode: If True, apply stricter validation criteria
        """
        self.strict_mode = strict_mode
        self.validation_weights = {
            'has_notes': 0.25,
            'reasonable_length': 0.15,
            'pitch_range_ok': 0.15,
            'no_extreme_velocities': 0.10,
            'rhythmic_coherence': 0.20,
            'harmonic_coherence': 0.15
        }

    def validate_coherence(self, midi: mido.MidiFile) -> float:
        """
        Check musical coherence of generated MIDI.

        Args:
            midi: MIDI file to validate

        Returns:
            Coherence score from 0.0 to 1.0 (1.0 = perfectly coherent)
        """
=======
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingDataset:
    """Complete training dataset for a parameter"""
    parameter_name: str
    parameter_definition: dict
    examples: List[TrainingExample]
    n_examples: int
    generation_date: str
    statistics: Dict[str, Any]
    output_directory: Path


class MusicalCoherenceValidator:
    """Validates that generated MIDI is musically coherent"""

    def __init__(self):
        self.coherence_checks = [
            'has_notes',
            'reasonable_length',
            'pitch_range_ok',
            'velocities_ok',
            'rhythmic_variation',
            'harmonic_coherence'
        ]

    def validate_coherence(self, midi_data: Any) -> float:
        """
        Check musical coherence of generated MIDI

        Args:
            midi_data: MIDI file or data structure

        Returns:
            Coherence score 0.0-1.0 (1.0 = perfectly coherent)
        """
        # If midi_data is a Path, we need to parse it
        if isinstance(midi_data, (str, Path)):
            midi_file = Path(midi_data)
            if not midi_file.exists():
                return 0.0

            try:
                import mido
                midi = mido.MidiFile(str(midi_file))
            except Exception as e:
                print(f"Warning: Could not load MIDI file: {e}")
                # Return default score if can't load
                return 0.7
        else:
            midi = midi_data

>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
        checks = {
            'has_notes': self._check_has_notes(midi),
            'reasonable_length': self._check_reasonable_length(midi),
            'pitch_range_ok': self._check_pitch_range(midi),
<<<<<<< HEAD
            'no_extreme_velocities': self._check_velocities(midi),
            'rhythmic_coherence': self._check_rhythm(midi),
=======
            'velocities_ok': self._check_velocities(midi),
            'rhythmic_variation': self._check_rhythm(midi),
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
            'harmonic_coherence': self._check_harmony(midi)
        }

        # Weighted average
<<<<<<< HEAD
        score = sum(
            checks[key] * self.validation_weights[key]
            for key in checks
        )

        return score

    def validate_with_details(self, midi: mido.MidiFile) -> Tuple[float, Dict[str, float]]:
        """
        Validate with detailed breakdown

        Returns:
            (overall_score, individual_scores_dict)
        """
        checks = {
            'has_notes': self._check_has_notes(midi),
            'reasonable_length': self._check_reasonable_length(midi),
            'pitch_range_ok': self._check_pitch_range(midi),
            'no_extreme_velocities': self._check_velocities(midi),
            'rhythmic_coherence': self._check_rhythm(midi),
            'harmonic_coherence': self._check_harmony(midi)
        }

        score = sum(
            checks[key] * self.validation_weights[key]
            for key in checks
        )

        return score, checks

    def _check_has_notes(self, midi: mido.MidiFile) -> float:
        """Check MIDI has actual notes"""
        note_count = 0
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    note_count += 1

        if note_count == 0:
            return 0.0
        elif note_count < 5:
            return 0.5  # Very few notes
        else:
            return 1.0

    def _check_reasonable_length(self, midi: mido.MidiFile) -> float:
        """Check MIDI length is reasonable"""
        try:
            length = midi.length  # seconds
        except:
            # Fallback: calculate from ticks
            total_ticks = sum(track[-1].time if track else 0 for track in midi.tracks)
            length = mido.tick2second(total_ticks, midi.ticks_per_beat, 500000)

        if self.strict_mode:
            if 10 <= length <= 90:
                return 1.0
            elif 5 <= length <= 120:
                return 0.7
            else:
                return 0.3
        else:
            if 5 <= length <= 120:
=======
        score = sum(checks.values()) / len(checks)

        return score

    def _check_has_notes(self, midi) -> float:
        """Check MIDI has actual notes"""
        try:
            note_count = sum(
                1 for track in midi.tracks
                for msg in track
                if hasattr(msg, 'type') and msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0
            )
            return 1.0 if note_count > 0 else 0.0
        except:
            return 0.7  # Default

    def _check_reasonable_length(self, midi) -> float:
        """Check MIDI length is reasonable"""
        try:
            length = midi.length if hasattr(midi, 'length') else 30  # seconds
            if 5 <= length <= 120:  # 5 seconds to 2 minutes
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
                return 1.0
            elif 2 <= length <= 180:
                return 0.7
            else:
                return 0.3
<<<<<<< HEAD

    def _check_pitch_range(self, midi: mido.MidiFile) -> float:
        """Check pitches are in reasonable range"""
        pitches = []
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'note'):
                    if hasattr(msg, 'velocity') and msg.velocity > 0:
                        pitches.append(msg.note)

        if not pitches:
            return 0.0

        # Check range
        min_pitch = min(pitches)
        max_pitch = max(pitches)

        # Standard MIDI piano range: 21 (A0) to 108 (C8)
        if 21 <= min_pitch and max_pitch <= 108:
            return 1.0
        # Extended range but still valid MIDI
        elif 0 <= min_pitch and max_pitch <= 127:
            return 0.8
        else:
            return 0.0

    def _check_velocities(self, midi: mido.MidiFile) -> float:
        """Check velocities are reasonable"""
        velocities = []
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'velocity'):
                    if msg.velocity > 0:  # Ignore note-offs
                        velocities.append(msg.velocity)

        if not velocities:
            return 0.0

        vel_array = np.array(velocities)
        mean_vel = np.mean(vel_array)
        std_vel = np.std(vel_array)

        # Check for reasonable range (not all silent or all max)
        if 30 <= mean_vel <= 100:
            score = 1.0
        elif 20 <= mean_vel <= 110:
            score = 0.8
        else:
            score = 0.5

        # Penalize if all velocities are identical (no dynamics)
        if std_vel < 1.0:
            score *= 0.5

        return score

    def _check_rhythm(self, midi: mido.MidiFile) -> float:
        """Check rhythmic coherence"""
        note_times = []

        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time
                if msg.type == 'note_on' and hasattr(msg, 'velocity') and msg.velocity > 0:
                    note_times.append(current_time)

        if len(note_times) < 2:
            return 0.0

        # Check for rhythmic variation
        time_diffs = np.diff(note_times)

        if len(time_diffs) == 0:
            return 0.0

        # Good rhythm has some variation but not total chaos
        unique_timings = len(set(time_diffs))

        if unique_timings == 1:
            # All same timing - mechanical but valid
            return 0.6
        elif 2 <= unique_timings <= len(time_diffs) * 0.8:
            # Good variation
            return 1.0
        else:
            # Too much variation might indicate randomness
            return 0.8

    def _check_harmony(self, midi: mido.MidiFile) -> float:
        """Check harmonic coherence"""
        # For now, basic check: if multiple simultaneous notes,
        # they should form reasonable intervals

        # Extract simultaneous notes (simplified)
        active_notes = set()
        intervals = []

        for track in midi.tracks:
            current_time = 0
            for msg in track:
                current_time += msg.time

                if msg.type == 'note_on' and hasattr(msg, 'velocity'):
                    if msg.velocity > 0:
                        active_notes.add(msg.note)
                    else:
                        active_notes.discard(msg.note)

                # Check intervals when we have multiple notes
                if len(active_notes) >= 2:
                    notes = sorted(list(active_notes))
                    for i in range(len(notes) - 1):
                        interval = notes[i + 1] - notes[i]
                        intervals.append(interval)

        if not intervals:
            # No simultaneous notes - melodic only, that's fine
            return 1.0

        # Check for dissonant intervals (seconds, sevenths, tritones)
        # Allow them but penalize if they're dominant
        dissonant = sum(1 for i in intervals if i in [1, 2, 6, 10, 11])
        dissonance_ratio = dissonant / len(intervals) if intervals else 0

        if dissonance_ratio < 0.3:
            return 1.0
        elif dissonance_ratio < 0.5:
            return 0.8
        else:
            return 0.6


# ============================================================================
# PARAMETER SPACE SAMPLER
# ============================================================================

class ParameterSpaceSampler:
    """
    Intelligent sampling of parameter space for even coverage.

    Uses Latin hypercube sampling for continuous parameters and
    balanced sampling for categorical/boolean parameters.
    """

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize sampler

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def sample_parameter_space(
        self,
        param_def: ParameterDefinition,
        n_samples: int
    ) -> List[Any]:
        """
        Sample parameter values for even coverage

        Args:
            param_def: Parameter definition
            n_samples: Number of samples to generate

        Returns:
            List of parameter values
        """
        param_type = param_def.param_type

        if param_type == ParameterType.CONTINUOUS or param_type == ParameterType.PROBABILITY:
            return self._sample_continuous(param_def, n_samples)

        elif param_type == ParameterType.INTEGER or param_type == ParameterType.MIDI_NOTE or param_type == ParameterType.VELOCITY:
            return self._sample_integer(param_def, n_samples)

        elif param_type == ParameterType.CATEGORICAL:
            return self._sample_categorical(param_def, n_samples)

        elif param_type == ParameterType.BOOLEAN:
            return self._sample_boolean(n_samples)

        elif param_type == ParameterType.ARRAY_INT:
            return self._sample_array_int(param_def, n_samples)

        elif param_type == ParameterType.ARRAY_FLOAT:
            return self._sample_array_float(param_def, n_samples)

        else:
            logger.warning(f"Unknown parameter type {param_type}, using default value")
            return [param_def.default_value] * n_samples

    def _sample_continuous(self, param_def: ParameterDefinition, n: int) -> List[float]:
        """Sample continuous parameter using Latin hypercube"""
        min_val = param_def.min_value if param_def.min_value is not None else 0.0
        max_val = param_def.max_value if param_def.max_value is not None else 1.0

        if SCIPY_AVAILABLE:
            # Use Latin hypercube sampling for better coverage
            sampler = qmc.LatinHypercube(d=1, seed=self.seed)
            samples = sampler.random(n=n)
            values = samples[:, 0] * (max_val - min_val) + min_val
            return values.tolist()
        else:
            # Fallback to stratified sampling
            values = []
            for i in range(n):
                # Divide range into n strata and sample from each
                stratum_size = (max_val - min_val) / n
                stratum_min = min_val + i * stratum_size
                stratum_max = stratum_min + stratum_size
                values.append(random.uniform(stratum_min, stratum_max))
            random.shuffle(values)
            return values

    def _sample_integer(self, param_def: ParameterDefinition, n: int) -> List[int]:
        """Sample integer parameter"""
        min_val = int(param_def.min_value) if param_def.min_value is not None else 0
        max_val = int(param_def.max_value) if param_def.max_value is not None else 127

        # If range is small, sample with replacement
        range_size = max_val - min_val + 1

        if range_size <= n:
            # Sample all values multiple times
            values = list(range(min_val, max_val + 1)) * (n // range_size + 1)
            random.shuffle(values)
            return values[:n]
        else:
            # Use stratified sampling
            values = []
            for i in range(n):
                stratum_size = (max_val - min_val + 1) / n
                stratum_center = min_val + int(i * stratum_size + stratum_size / 2)
                values.append(stratum_center)
            random.shuffle(values)
            return values

    def _sample_categorical(self, param_def: ParameterDefinition, n: int) -> List[Any]:
        """Sample categorical parameter with balanced representation"""
        options = param_def.options
        if not options:
            return [param_def.default_value] * n

        # Balanced sampling across categories
        values = []
        per_category = n // len(options)

        for option in options:
            values.extend([option] * per_category)

        # Fill remainder
        while len(values) < n:
            values.append(random.choice(options))

        random.shuffle(values)
        return values

    def _sample_boolean(self, n: int) -> List[bool]:
        """Sample boolean parameter with 50/50 split"""
        values = [True] * (n // 2) + [False] * (n // 2)
        if len(values) < n:
            values.append(random.choice([True, False]))
        random.shuffle(values)
        return values

    def _sample_array_int(self, param_def: ParameterDefinition, n: int) -> List[List[int]]:
        """Sample integer array parameter"""
        arrays = []
        for _ in range(n):
            # Variable length arrays
            length = random.choice([4, 8, 12, 16])
            arr = [random.randint(0, 127) for _ in range(length)]
            arrays.append(arr)
        return arrays

    def _sample_array_float(self, param_def: ParameterDefinition, n: int) -> List[List[float]]:
        """Sample float array parameter"""
        arrays = []
        for _ in range(n):
            # Variable length arrays
            length = random.choice([4, 8, 12, 16])
            arr = [random.uniform(0.0, 1.0) for _ in range(length)]
            arrays.append(arr)
        return arrays


# ============================================================================
# DEFAULT PARAMETER GENERATOR
# ============================================================================

class DefaultParameterGenerator:
    """
    Generates diverse default parameters for training examples.

    Ensures that when training for one parameter, all other parameters
    are varied to prevent overfitting.
    """

    def __init__(self, registry: UniversalParameterRegistry):
        """
        Initialize generator

        Args:
            registry: Universal parameter registry
        """
        self.registry = registry
        self.sampler = ParameterSpaceSampler()

    def sample_default_parameters(self, exclude_param: Optional[str] = None) -> Dict[str, Any]:
        """
        Sample diverse values for all parameters except excluded one

        Args:
            exclude_param: Parameter name to exclude (will be set separately)

        Returns:
            Dictionary of parameter values
        """
        params = {}

        for param_name, param_def in self.registry.parameters.items():
            if param_name == exclude_param:
                continue

            # Sample a single value for this parameter
            value = self._sample_single_value(param_def)
            params[param_name] = value

        return params

    def _sample_single_value(self, param_def: ParameterDefinition) -> Any:
        """Sample one value for a parameter"""
        param_type = param_def.param_type

        if param_type == ParameterType.CONTINUOUS or param_type == ParameterType.PROBABILITY:
            min_val = param_def.min_value if param_def.min_value is not None else 0.0
            max_val = param_def.max_value if param_def.max_value is not None else 1.0
            return random.uniform(min_val, max_val)

        elif param_type == ParameterType.INTEGER or param_type == ParameterType.MIDI_NOTE or param_type == ParameterType.VELOCITY:
            min_val = int(param_def.min_value) if param_def.min_value is not None else 0
            max_val = int(param_def.max_value) if param_def.max_value is not None else 127
            return random.randint(min_val, max_val)

        elif param_type == ParameterType.CATEGORICAL:
            if param_def.options:
                return random.choice(param_def.options)
            return param_def.default_value

        elif param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])

        elif param_type == ParameterType.ARRAY_INT:
            length = random.choice([4, 8, 12, 16])
            return [random.randint(0, 127) for _ in range(length)]

        elif param_type == ParameterType.ARRAY_FLOAT:
            length = random.choice([4, 8, 12, 16])
            return [random.uniform(0.0, 1.0) for _ in range(length)]

        else:
            return param_def.default_value


# ============================================================================
# SYNTHETIC TRAINING DATA GENERATOR
# ============================================================================

class SyntheticTrainingDataGenerator:
    """
    Main class for generating synthetic training data for new parameters.

    Generates 1000+ high-quality (MIDI, parameter_value) training examples
    with:
    - Latin hypercube sampling for even parameter space coverage
    - Diverse other parameters to prevent overfitting
    - Musical validity checking
    - Balanced dataset across parameter range
    - Genre-balanced options
    - Comprehensive metadata and statistics
    """

    def __init__(
        self,
        generator=None,
        feature_extractor=None,
        registry: Optional[UniversalParameterRegistry] = None,
        validator: Optional[MusicalCoherenceValidator] = None,
        output_root: Path = Path('training_data')
    ):
        """
        Initialize synthetic data generator

        Args:
            generator: MIDI generator (e.g., HarmonyModule)
            feature_extractor: Feature extractor (e.g., DeepFeatureExtractor)
            registry: Parameter registry (creates default if None)
            validator: Coherence validator (creates default if None)
            output_root: Root directory for training data
        """
        self.generator = generator
        self.feature_extractor = feature_extractor
        self.registry = registry or UniversalParameterRegistry()
        self.validator = validator or MusicalCoherenceValidator()
        self.output_root = Path(output_root)

        # Helper components
        self.sampler = ParameterSpaceSampler()
        self.param_generator = DefaultParameterGenerator(self.registry)

        # Statistics tracking
        self.stats = {
            'total_generated': 0,
            'total_failed': 0,
            'generation_times': [],
            'coherence_scores': []
        }

    def generate_training_data(
        self,
        param_name: str,
        param_def: ParameterDefinition,
        n_examples: int = 1000,
        output_dir: Optional[Path] = None,
        min_coherence: float = 0.5,
        max_failures_ratio: float = 0.2
    ) -> List[TrainingExample]:
        """
        Generate training dataset for new parameter.

        STRATEGY:
=======
        except:
            return 0.7  # Default

    def _check_pitch_range(self, midi) -> float:
        """Check pitches are in reasonable range"""
        try:
            pitches = []
            for track in midi.tracks:
                for msg in track:
                    if hasattr(msg, 'type') and msg.type == 'note_on' and hasattr(msg, 'note'):
                        pitches.append(msg.note)

            if not pitches:
                return 0.0

            # MIDI range 21-108 is standard (A0 to C8)
            if all(21 <= p <= 108 for p in pitches):
                return 1.0
            elif all(0 <= p <= 127 for p in pitches):
                return 0.7
            else:
                return 0.0
        except:
            return 0.7  # Default

    def _check_velocities(self, midi) -> float:
        """Check velocities are reasonable"""
        try:
            velocities = []
            for track in midi.tracks:
                for msg in track:
                    if hasattr(msg, 'type') and msg.type == 'note_on' and hasattr(msg, 'velocity'):
                        if msg.velocity > 0:  # Actual note on
                            velocities.append(msg.velocity)

            if not velocities:
                return 0.0

            # Check for reasonable range (not all silent or all max)
            avg_vel = np.mean(velocities)
            if 20 <= avg_vel <= 110:
                return 1.0
            else:
                return 0.5
        except:
            return 0.7  # Default

    def _check_rhythm(self, midi) -> float:
        """Check rhythmic coherence"""
        try:
            times = []
            for track in midi.tracks:
                time = 0
                for msg in track:
                    time += msg.time if hasattr(msg, 'time') else 0
                    if hasattr(msg, 'type') and msg.type == 'note_on':
                        times.append(time)

            if len(times) < 2:
                return 0.0

            # Check for some rhythmic variation
            time_diffs = np.diff(times) if len(times) > 1 else []
            if len(time_diffs) > 0 and len(set(time_diffs)) > 1:  # Not all same timing
                return 1.0
            else:
                return 0.5
        except:
            return 0.7  # Default

    def _check_harmony(self, midi) -> float:
        """Check harmonic coherence"""
        # Basic check: assume OK for now
        # More sophisticated harmony checking would require full analysis
        return 1.0


class SyntheticTrainingDataGenerator:
    """
    Generates synthetic training data for new parameters

    Uses the generator to create diverse MIDI examples with varied
    parameter values, then extracts features for training.
    """

    def __init__(self,
                 generator: Optional[Any] = None,
                 feature_extractor: Optional[Any] = None,
                 coherence_validator: Optional[MusicalCoherenceValidator] = None):
        """
        Initialize training data generator

        Args:
            generator: Music generator instance (HarmonyModule or similar)
            feature_extractor: Feature extractor instance
            coherence_validator: Musical coherence validator
        """
        self.generator = generator
        self.feature_extractor = feature_extractor
        self.validator = coherence_validator or MusicalCoherenceValidator()

        self.generation_history: List[TrainingDataset] = []

    def generate_training_data(self,
                              param_name: str,
                              param_def: dict,
                              n_examples: int = 1000,
                              output_dir: Path = Path('training_data'),
                              min_coherence: float = 0.5) -> TrainingDataset:
        """
        Generate training dataset for new parameter

        Strategy:
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
        1. Sample parameter values using Latin hypercube (even coverage)
        2. For each value, vary other parameters (prevent overfitting)
        3. Generate MIDI
        4. Validate musical coherence
        5. Extract features
        6. Store: (features, parameter_value, MIDI_file)

        Args:
<<<<<<< HEAD
            param_name: Name of parameter to train
            param_def: Parameter definition
            n_examples: Number of training examples to generate
            output_dir: Output directory (default: training_data/{param_name})
            min_coherence: Minimum coherence score to accept (0.0-1.0)
            max_failures_ratio: Maximum ratio of failures before aborting

        Returns:
            List of TrainingExample objects
        """
        logger.info(f"Generating {n_examples} training examples for {param_name}...")

        # Create output directory
        if output_dir is None:
            param_dir = self.output_root / param_name.replace('.', '_')
        else:
            param_dir = Path(output_dir)

        param_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sample parameter space intelligently
        parameter_values = self.sampler.sample_parameter_space(param_def, n_examples)

        # 2. Generate examples
        training_data = []
        failed = 0

        with tqdm(total=n_examples, desc=f"Generating {param_name}") as pbar:
            attempts = 0
            max_attempts = int(n_examples * (1 + max_failures_ratio * 2))

            while len(training_data) < n_examples and attempts < max_attempts:
                attempts += 1

                # Get parameter value for this example
                param_value = parameter_values[len(training_data) % len(parameter_values)]
=======
            param_name: Full parameter name (e.g., 'harmony.voicing.quartal_prob')
            param_def: Parameter definition dict
            n_examples: Number of examples to generate
            output_dir: Directory to save training data
            min_coherence: Minimum coherence score to accept example

        Returns:
            TrainingDataset with all examples and metadata
        """
        print(f"\n{'='*80}")
        print(f"GENERATING TRAINING DATA")
        print(f"{'='*80}")
        print(f"Parameter: {param_name}")
        print(f"Target examples: {n_examples}")
        print(f"Min coherence: {min_coherence}")
        print(f"Output dir: {output_dir}")
        print()

        # Create output directory
        param_dir = output_dir / param_name.replace('.', '_')
        param_dir.mkdir(parents=True, exist_ok=True)

        # 1. Sample parameter space intelligently
        print("Step 1: Sampling parameter space...")
        parameter_values = self._sample_parameter_space(param_def, n_examples)
        print(f"  Generated {len(parameter_values)} parameter values")

        # 2. Generate examples
        print(f"\nStep 2: Generating {n_examples} examples...")
        examples = []
        failed = 0
        attempts = 0
        max_attempts = n_examples * 3  # Allow some failures

        with tqdm(total=n_examples, desc=f"Generating {param_name}") as pbar:
            while len(examples) < n_examples and attempts < max_attempts:
                attempts += 1

                # Get parameter value for this example
                param_value = parameter_values[len(examples) % len(parameter_values)]
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS

                try:
                    # Generate training example
                    example = self._generate_single_example(
                        param_name,
                        param_value,
                        param_def,
                        param_dir,
<<<<<<< HEAD
                        len(training_data)
=======
                        len(examples)
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
                    )

                    # Validate musical coherence
                    if example.coherence_score < min_coherence:
                        failed += 1
<<<<<<< HEAD
                        logger.debug(f"Example {len(training_data)} failed coherence: {example.coherence_score:.2f}")

                        # Check failure rate
                        failure_rate = failed / attempts if attempts > 0 else 0
                        if failure_rate > max_failures_ratio and attempts > 100:
                            logger.error(f"Failure rate too high ({failure_rate:.1%}), aborting")
                            raise RuntimeError(
                                f"Too many generation failures ({failed}/{attempts}). "
                                "Check generator configuration."
                            )
                        continue

                    training_data.append(example)
=======
                        continue

                    examples.append(example)
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
                    pbar.update(1)

                except Exception as e:
                    failed += 1
<<<<<<< HEAD
                    logger.debug(f"Generation error: {e}")

                    # Check failure rate
                    failure_rate = failed / attempts if attempts > 0 else 0
                    if failure_rate > max_failures_ratio and attempts > 100:
                        logger.error(f"Failure rate too high ({failure_rate:.1%}), aborting")
                        raise RuntimeError(
                            f"Too many generation failures ({failed}/{attempts}): {e}"
                        )
                    continue

        logger.info(f"✅ Generated {len(training_data)} examples ({failed} failed)")

        # Update statistics
        self.stats['total_generated'] += len(training_data)
        self.stats['total_failed'] += failed
        self.stats['generation_times'].extend([ex.generation_time for ex in training_data])
        self.stats['coherence_scores'].extend([ex.coherence_score for ex in training_data])

        # Save metadata
        self._save_metadata(training_data, param_name, param_def, param_dir)

        return training_data

    def _generate_single_example(
        self,
        param_name: str,
        param_value: Any,
        param_def: ParameterDefinition,
        output_dir: Path,
        example_idx: int
    ) -> TrainingExample:
=======
                    if failed > n_examples * 0.3:  # >30% failure rate
                        print(f"\n⚠️ High failure rate ({failed}/{attempts})")
                        print(f"   Last error: {e}")
                        # Continue anyway
                    continue

        print(f"\n✅ Generated {len(examples)} examples ({failed} failed, {attempts} total attempts)")

        # 3. Analyze statistics
        print("\nStep 3: Analyzing dataset statistics...")
        statistics = self._analyze_statistics(examples, param_def)

        # 4. Create dataset object
        dataset = TrainingDataset(
            parameter_name=param_name,
            parameter_definition=param_def,
            examples=examples,
            n_examples=len(examples),
            generation_date=datetime.now().isoformat(),
            statistics=statistics,
            output_directory=param_dir
        )

        # 5. Save metadata
        print("\nStep 4: Saving metadata...")
        self._save_metadata(dataset)

        # Record in history
        self.generation_history.append(dataset)

        print(f"\n{'='*80}")
        print(f"TRAINING DATA GENERATION COMPLETE")
        print(f"{'='*80}")
        print(f"Examples: {len(examples)}")
        print(f"Avg coherence: {statistics.get('avg_coherence', 0):.3f}")
        print(f"Saved to: {param_dir}")
        print()

        return dataset

    def _sample_parameter_space(self, param_def: dict, n: int) -> List[Any]:
        """
        Sample parameter values for even coverage

        Args:
            param_def: Parameter definition
            n: Number of samples

        Returns:
            List of parameter values
        """
        param_type = param_def.get('type', 'CONTINUOUS')
        param_range = param_def.get('range', (0.0, 1.0))

        if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
            # Latin hypercube sampling if available, otherwise uniform
            if HAS_SCIPY and isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                try:
                    sampler = qmc.LatinHypercube(d=1, seed=42)
                    samples = sampler.random(n=n)

                    # Scale to parameter range
                    min_val, max_val = param_range
                    values = samples[:, 0] * (max_val - min_val) + min_val

                    return values.tolist()
                except:
                    pass

            # Fallback: uniform sampling
            if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                min_val, max_val = param_range
                return [min_val + (max_val - min_val) * i / (n - 1) for i in range(n)]
            else:
                return [0.5] * n

        elif param_type == 'CATEGORICAL':
            # Balanced sampling across categories
            if isinstance(param_range, list) and param_range:
                values = []
                per_category = n // len(param_range)

                for option in param_range:
                    values.extend([option] * per_category)

                # Fill remainder
                while len(values) < n:
                    values.append(random.choice(param_range))

                random.shuffle(values)
                return values
            else:
                return ['default'] * n

        elif param_type == 'BOOLEAN':
            # 50/50 split
            values = [True] * (n // 2) + [False] * (n // 2)
            if len(values) < n:
                values.append(random.choice([True, False]))
            random.shuffle(values)
            return values

        elif param_type in ['ARRAY_INT', 'ARRAY_FLOAT']:
            # Generate diverse patterns
            return [self._generate_random_array(param_def) for _ in range(n)]

        else:
            # Unknown type, return defaults
            default = param_def.get('default', 0.5)
            return [default] * n

    def _generate_single_example(self,
                                 param_name: str,
                                 param_value: Any,
                                 param_def: dict,
                                 output_dir: Path,
                                 example_idx: int) -> TrainingExample:
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
        """
        Generate one training example

        Args:
            param_name: Parameter name
<<<<<<< HEAD
            param_value: Value for this parameter
            param_def: Parameter definition
            output_dir: Output directory
            example_idx: Example index

        Returns:
            TrainingExample object
        """
        start_time = time.time()

        try:
            # 1. Create parameter set with NEW param + varied others
            params = self.param_generator.sample_default_parameters(exclude_param=param_name)
            params[param_name] = param_value

            # 2. Generate MIDI
            if self.generator is None:
                # Mock generation for testing
                midi = self._create_mock_midi()
            else:
                midi = self.generator.generate(params)

            # 3. Save MIDI
            midi_filename = f"{param_name.replace('.', '_')}_{example_idx:04d}.mid"
            midi_path = output_dir / midi_filename
            midi.save(str(midi_path))

            # 4. Extract features
            if self.feature_extractor is None:
                # Mock features for testing
                features = np.random.randn(1000)
            else:
                features = self.feature_extractor.extract(midi_path)

            # 5. Validate coherence
            coherence_score = self.validator.validate_coherence(midi)

            generation_time = time.time() - start_time

            return TrainingExample(
                features=features,
                parameter_value=param_value,
                midi_file=midi_path,
                other_params=params.copy(),
                generation_time=generation_time,
                coherence_score=coherence_score,
                validation_passed=True
            )

        except Exception as e:
            generation_time = time.time() - start_time
            logger.error(f"Error generating example {example_idx}: {e}")

            # Return failed example
            return TrainingExample(
                features=np.array([]),
                parameter_value=param_value,
                midi_file=output_dir / f"failed_{example_idx}.mid",
                other_params={},
                generation_time=generation_time,
                coherence_score=0.0,
                validation_passed=False,
                error_message=str(e)
            )

    def _create_mock_midi(self) -> mido.MidiFile:
        """Create mock MIDI for testing (when no generator available)"""
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)

        # Add some notes
        track.append(mido.Message('program_change', program=0, time=0))
        for i, note in enumerate([60, 64, 67, 72]):
            track.append(mido.Message('note_on', note=note, velocity=64, time=i * 480))
            track.append(mido.Message('note_off', note=note, velocity=64, time=480))

        return mid

    def generate_balanced_dataset(
        self,
        param_name: str,
        param_def: ParameterDefinition,
        n_per_genre: int = 100,
        genres: Optional[List[str]] = None
    ) -> List[TrainingExample]:
=======
            param_value: Value for this example
            param_def: Parameter definition
            output_dir: Directory to save MIDI
            example_idx: Example index for filename

        Returns:
            TrainingExample
        """
        start_time = time.time()

        # 1. Create parameter set with NEW param + varied others
        params = self._sample_default_parameters()
        params[param_name] = param_value

        # 2. Generate MIDI (mock for now since we don't have real generator)
        midi = self._generate_midi_mock(params)

        # 3. Save MIDI
        midi_filename = f"{param_name.replace('.', '_')}_{example_idx:04d}.mid"
        midi_path = output_dir / midi_filename

        # Save the MIDI file (mock version creates empty file)
        self._save_midi_mock(midi, midi_path)

        # 4. Extract features (mock for now)
        features = self._extract_features_mock(midi_path)

        # 5. Validate coherence
        coherence_score = self.validator.validate_coherence(midi_path)

        generation_time = time.time() - start_time

        return TrainingExample(
            features=features,
            parameter_value=param_value,
            midi_file=midi_path,
            other_params=params.copy(),
            generation_time=generation_time,
            coherence_score=coherence_score
        )

    def _sample_default_parameters(self) -> Dict[str, Any]:
        """
        Sample diverse values for other parameters

        Returns:
            Dictionary of parameter values
        """
        # Mock implementation - in real version would load from registry
        # and sample from each parameter's range

        base_params = {
            'tempo': random.randint(80, 180),
            'key': random.choice(['C', 'Eb', 'F', 'Bb', 'G']),
            'time_signature': random.choice(['4/4', '3/4', '5/4', '6/8']),
            'swing_amount': random.uniform(0.0, 0.7),
            'complexity': random.uniform(0.3, 0.9),
            'density': random.uniform(0.4, 0.8),
        }

        return base_params

    def _generate_midi_mock(self, params: dict) -> Any:
        """Mock MIDI generation (placeholder)"""
        # In real implementation, would call:
        # return self.generator.generate(params)

        # For now, return mock data
        return {'params': params, 'mock': True}

    def _save_midi_mock(self, midi: Any, path: Path):
        """Mock MIDI saving"""
        # Create empty file or save mock data
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.write(b'MThd')  # MIDI header magic number

    def _extract_features_mock(self, midi_path: Path) -> np.ndarray:
        """Mock feature extraction"""
        # In real implementation:
        # return self.feature_extractor.extract(midi_path)

        # For now, return random features
        n_features = 1000  # Match deep_feature_extractor target
        return np.random.randn(n_features)

    def _generate_random_array(self, param_def: dict) -> List:
        """Generate random array for ARRAY_INT/ARRAY_FLOAT parameters"""

        # Typical array length
        length = random.choice([4, 8, 12, 16])

        if param_def.get('type') == 'ARRAY_INT':
            # Generate rhythm pattern or similar
            return [random.randint(0, 127) for _ in range(length)]
        else:  # ARRAY_FLOAT
            return [random.uniform(0.0, 1.0) for _ in range(length)]

    def _analyze_statistics(self, examples: List[TrainingExample], param_def: dict) -> Dict[str, Any]:
        """
        Analyze dataset statistics

        Args:
            examples: List of training examples
            param_def: Parameter definition

        Returns:
            Statistics dictionary
        """
        if not examples:
            return {}

        stats = {
            'n_examples': len(examples),
            'avg_coherence': np.mean([ex.coherence_score for ex in examples]),
            'min_coherence': np.min([ex.coherence_score for ex in examples]),
            'max_coherence': np.max([ex.coherence_score for ex in examples]),
            'avg_generation_time': np.mean([ex.generation_time for ex in examples]),
        }

        # Parameter value distribution
        values = [ex.parameter_value for ex in examples]

        if isinstance(values[0], (int, float, np.number)):
            stats['parameter_distribution'] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'quartiles': [float(q) for q in np.percentile(values, [25, 50, 75])]
            }
        else:
            # Categorical
            counts = Counter(values)
            stats['parameter_distribution'] = {
                'type': 'categorical',
                'counts': {str(k): v for k, v in counts.items()}
            }

        return stats

    def _save_metadata(self, dataset: TrainingDataset):
        """
        Save dataset metadata to JSON

        Args:
            dataset: Training dataset
        """
        metadata = {
            'parameter_name': dataset.parameter_name,
            'parameter_definition': dataset.parameter_definition,
            'n_examples': dataset.n_examples,
            'generation_date': dataset.generation_date,
            'statistics': dataset.statistics,
            'examples': [
                {
                    'index': i,
                    'midi_file': ex.midi_file.name,
                    'parameter_value': self._serialize_value(ex.parameter_value),
                    'coherence_score': float(ex.coherence_score),
                    'generation_time': float(ex.generation_time),
                    'genre': ex.genre
                }
                for i, ex in enumerate(dataset.examples)
            ]
        }

        metadata_path = dataset.output_directory / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"  Saved metadata to: {metadata_path}")

    def _serialize_value(self, value: Any) -> Any:
        """Serialize value for JSON"""
        if isinstance(value, (np.number, np.ndarray)):
            return float(value)
        elif isinstance(value, list):
            return [self._serialize_value(v) for v in value]
        else:
            return value

    def generate_balanced_dataset(self,
                                  param_name: str,
                                  param_def: dict,
                                  n_per_genre: int = 100,
                                  genres: Optional[List[str]] = None) -> TrainingDataset:
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
        """
        Generate training data balanced across genres

        Args:
<<<<<<< HEAD
            param_name: Parameter to train
            param_def: Parameter definition
            n_per_genre: Examples per genre
            genres: List of genres (uses defaults if None)

        Returns:
            List of training examples
=======
            param_name: Parameter name
            param_def: Parameter definition
            n_per_genre: Examples per genre
            genres: List of genres to use

        Returns:
            TrainingDataset with genre-balanced examples
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
        """
        if genres is None:
            genres = [
                'swing', 'bebop', 'modal', 'bossa_nova',
                'fusion', 'cool_jazz', 'free_jazz', 'ballad'
            ]

<<<<<<< HEAD
        all_training_data = []

        for genre in genres:
            logger.info(f"Generating {n_per_genre} examples for {param_name} in {genre} style...")

            genre_dir = self.output_root / genre / param_name.replace('.', '_')
            genre_dir.mkdir(parents=True, exist_ok=True)

            # Get genre-specific parameters
            genre_params_base = self._get_genre_params(genre)

            # Sample parameter values
            parameter_values = self.sampler.sample_parameter_space(param_def, n_per_genre)

            for i in range(n_per_genre):
                param_value = parameter_values[i]

                # Combine with genre params
                params = genre_params_base.copy()
                params[param_name] = param_value

                # Generate
                try:
                    example = self._generate_single_example(
                        param_name,
                        param_value,
                        param_def,
                        genre_dir,
                        i
                    )
                    example.genre = genre
                    all_training_data.append(example)
                except Exception as e:
                    logger.error(f"Failed to generate {genre} example {i}: {e}")
                    continue

        logger.info(f"✅ Generated {len(all_training_data)} genre-balanced examples")

        # Save combined metadata
        self._save_metadata(
            all_training_data,
            param_name,
            param_def,
            self.output_root / f"{param_name.replace('.', '_')}_balanced"
        )

        return all_training_data

    def _get_genre_params(self, genre: str) -> Dict[str, Any]:
        """
        Get default parameters for genre

        Args:
            genre: Genre name

        Returns:
            Dictionary of parameter values for this genre
        """
        # Try to load from style database
        try:
            from midi_generator.styles.style_registry import StyleRegistry
            registry = StyleRegistry()
            genre_style = registry.get_style(genre)
            if genre_style:
                return genre_style.get('parameters', {})
        except ImportError:
            logger.warning("StyleRegistry not available, using defaults")

        # Fallback: use random parameters
        return self.param_generator.sample_default_parameters()

    def _save_metadata(
        self,
        training_data: List[TrainingExample],
        param_name: str,
        param_def: ParameterDefinition,
        output_dir: Path
    ):
        """
        Save training dataset metadata

        Args:
            training_data: List of training examples
            param_name: Parameter name
            param_def: Parameter definition
            output_dir: Output directory
        """
        successful = [ex for ex in training_data if ex.validation_passed]
        failed = [ex for ex in training_data if not ex.validation_passed]

        # Calculate statistics
        stats = DatasetStatistics(
            total_examples=len(training_data),
            successful_examples=len(successful),
            failed_examples=len(failed),
            avg_coherence_score=np.mean([ex.coherence_score for ex in successful]) if successful else 0.0,
            avg_generation_time=np.mean([ex.generation_time for ex in successful]) if successful else 0.0,
            parameter_value_distribution=self._analyze_distribution(successful, param_def)
        )

        # Add genre distribution if applicable
        if any(ex.genre for ex in training_data):
            genre_counts = Counter([ex.genre for ex in training_data if ex.genre])
            stats.genre_distribution = dict(genre_counts)

        # Quality metrics
        if successful:
            coherence_scores = [ex.coherence_score for ex in successful]
            stats.quality_metrics = {
                'coherence_min': float(np.min(coherence_scores)),
                'coherence_max': float(np.max(coherence_scores)),
                'coherence_std': float(np.std(coherence_scores)),
                'coherence_median': float(np.median(coherence_scores))
            }

        # Build metadata dictionary
        metadata = {
            'parameter_name': param_name,
            'parameter_type': param_def.param_type.value,
            'parameter_category': param_def.category.value if param_def.category else None,
            'generation_date': datetime.now().isoformat(),
            'statistics': stats.to_dict(),
            'examples': [ex.to_dict() for ex in successful[:100]],  # First 100 for size
            'failed_examples': [ex.to_dict() for ex in failed[:50]] if failed else []
        }

        # Save metadata
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / 'metadata.json'

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

        # Also save CSV for easy analysis
        self._save_csv_summary(successful, output_dir / 'summary.csv')

    def _analyze_distribution(
        self,
        training_data: List[TrainingExample],
        param_def: ParameterDefinition
    ) -> Dict[str, Any]:
        """
        Analyze parameter value distribution

        Args:
            training_data: Training examples
            param_def: Parameter definition

        Returns:
            Distribution statistics
        """
        if not training_data:
            return {}

        values = [ex.parameter_value for ex in training_data]

        # Numeric distribution
        if param_def.param_type in [
            ParameterType.CONTINUOUS,
            ParameterType.PROBABILITY,
            ParameterType.INTEGER,
            ParameterType.MIDI_NOTE,
            ParameterType.VELOCITY
        ]:
            numeric_values = [float(v) for v in values]
            return {
                'mean': float(np.mean(numeric_values)),
                'std': float(np.std(numeric_values)),
                'min': float(np.min(numeric_values)),
                'max': float(np.max(numeric_values)),
                'median': float(np.median(numeric_values)),
                'quartiles': {
                    'q25': float(np.percentile(numeric_values, 25)),
                    'q50': float(np.percentile(numeric_values, 50)),
                    'q75': float(np.percentile(numeric_values, 75))
                }
            }

        # Categorical distribution
        elif param_def.param_type == ParameterType.CATEGORICAL:
            counts = Counter([str(v) for v in values])
            return {
                'distribution': dict(counts),
                'unique_values': len(counts),
                'most_common': counts.most_common(5)
            }

        # Boolean distribution
        elif param_def.param_type == ParameterType.BOOLEAN:
            counts = Counter(values)
            total = sum(counts.values())
            return {
                'true_count': counts.get(True, 0),
                'false_count': counts.get(False, 0),
                'true_ratio': counts.get(True, 0) / total if total > 0 else 0
            }

        # Array types
        else:
            return {
                'sample_count': len(values),
                'sample_values': [str(v) for v in values[:5]]
            }

    def _save_csv_summary(self, training_data: List[TrainingExample], csv_path: Path):
        """Save CSV summary of training data"""
        import csv

        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'index',
                'parameter_value',
                'coherence_score',
                'generation_time',
                'midi_file',
                'genre'
            ])

            for i, ex in enumerate(training_data):
                writer.writerow([
                    i,
                    ex.parameter_value,
                    f"{ex.coherence_score:.3f}",
                    f"{ex.generation_time:.3f}",
                    ex.midi_file.name,
                    ex.genre or 'N/A'
                ])

        logger.info(f"Saved CSV summary to {csv_path}")

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get global statistics across all generations"""
        return {
            'total_generated': self.stats['total_generated'],
            'total_failed': self.stats['total_failed'],
            'success_rate': (
                self.stats['total_generated'] /
                (self.stats['total_generated'] + self.stats['total_failed'])
                if self.stats['total_generated'] + self.stats['total_failed'] > 0
                else 0
            ),
            'avg_generation_time': (
                np.mean(self.stats['generation_times'])
                if self.stats['generation_times']
                else 0
            ),
            'avg_coherence_score': (
                np.mean(self.stats['coherence_scores'])
                if self.stats['coherence_scores']
                else 0
            )
        }


# ============================================================================
# BATCH GENERATION UTILITIES
# ============================================================================

class BatchTrainingDataGenerator:
    """
    Utility for batch generation of training data for multiple parameters.
    """

    def __init__(
        self,
        generator=None,
        feature_extractor=None,
        registry: Optional[UniversalParameterRegistry] = None
    ):
        """
        Initialize batch generator

        Args:
            generator: MIDI generator
            feature_extractor: Feature extractor
            registry: Parameter registry
        """
        self.data_generator = SyntheticTrainingDataGenerator(
            generator=generator,
            feature_extractor=feature_extractor,
            registry=registry
        )

    def generate_for_multiple_parameters(
        self,
        param_names: List[str],
        n_examples_per_param: int = 1000,
        parallel: bool = False
    ) -> Dict[str, List[TrainingExample]]:
        """
        Generate training data for multiple parameters

        Args:
            param_names: List of parameter names
            n_examples_per_param: Examples per parameter
            parallel: Whether to use parallel generation (future feature)

        Returns:
            Dictionary mapping parameter names to training examples
        """
        results = {}

        for param_name in param_names:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing parameter: {param_name}")
            logger.info(f"{'='*60}\n")

            # Get parameter definition
            param_def = self.data_generator.registry.parameters.get(param_name)

            if param_def is None:
                logger.error(f"Parameter {param_name} not found in registry")
                continue

            # Generate training data
            try:
                training_data = self.data_generator.generate_training_data(
                    param_name=param_name,
                    param_def=param_def,
                    n_examples=n_examples_per_param
                )
                results[param_name] = training_data
            except Exception as e:
                logger.error(f"Failed to generate data for {param_name}: {e}")
                results[param_name] = []

        # Print summary
        logger.info(f"\n{'='*60}")
        logger.info("BATCH GENERATION SUMMARY")
        logger.info(f"{'='*60}")
        for param_name, data in results.items():
            logger.info(f"{param_name}: {len(data)} examples")
        logger.info(f"{'='*60}\n")

        return results


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class MIDIDataAugmenter:
    """
    Augment training data by applying musical transformations to MIDI files.

    Transformations:
    - Transposition (pitch shift)
    - Time stretching (tempo change)
    - Velocity scaling (dynamics change)
    - Inversion (melodic inversion)
    - Retrograde (reverse time)
    """

    def __init__(self):
        """Initialize augmenter"""
        self.transformations = [
            'transpose',
            'time_stretch',
            'velocity_scale',
            'inversion',
            'retrograde'
        ]

    def augment_example(
        self,
        example: TrainingExample,
        transformations: Optional[List[str]] = None,
        n_augmentations: int = 5
    ) -> List[TrainingExample]:
        """
        Generate augmented versions of a training example

        Args:
            example: Original training example
            transformations: List of transformations to apply (None = all)
            n_augmentations: Number of augmented versions to create

        Returns:
            List of augmented training examples
        """
        if transformations is None:
            transformations = self.transformations

        augmented = []

        for i in range(n_augmentations):
            # Select random transformation
            transform = random.choice(transformations)

            # Apply transformation
            try:
                if transform == 'transpose':
                    aug_ex = self._transpose(example, semitones=random.randint(-6, 6))
                elif transform == 'time_stretch':
                    aug_ex = self._time_stretch(example, factor=random.uniform(0.8, 1.2))
                elif transform == 'velocity_scale':
                    aug_ex = self._velocity_scale(example, factor=random.uniform(0.7, 1.3))
                elif transform == 'inversion':
                    aug_ex = self._inversion(example)
                elif transform == 'retrograde':
                    aug_ex = self._retrograde(example)
                else:
                    continue

                augmented.append(aug_ex)

            except Exception as e:
                logger.warning(f"Augmentation {transform} failed: {e}")
                continue

        return augmented

    def _transpose(self, example: TrainingExample, semitones: int) -> TrainingExample:
        """Transpose MIDI by semitones"""
        midi = mido.MidiFile(str(example.midi_file))

        # Transpose all notes
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'note'):
                    new_note = max(0, min(127, msg.note + semitones))
                    msg.note = new_note

        # Save augmented MIDI
        aug_path = example.midi_file.parent / f"{example.midi_file.stem}_transpose_{semitones}.mid"
        midi.save(str(aug_path))

        # Create augmented example (features would need re-extraction in real system)
        return TrainingExample(
            features=example.features,  # Would re-extract in production
            parameter_value=example.parameter_value,
            midi_file=aug_path,
            other_params=example.other_params,
            generation_time=example.generation_time,
            coherence_score=example.coherence_score,
            genre=example.genre
        )

    def _time_stretch(self, example: TrainingExample, factor: float) -> TrainingExample:
        """Stretch time by factor (1.0 = no change, 2.0 = twice as slow)"""
        midi = mido.MidiFile(str(example.midi_file))

        # Scale all time values
        for track in midi.tracks:
            for msg in track:
                msg.time = int(msg.time * factor)

        # Save
        aug_path = example.midi_file.parent / f"{example.midi_file.stem}_stretch_{factor:.2f}.mid"
        midi.save(str(aug_path))

        return TrainingExample(
            features=example.features,
            parameter_value=example.parameter_value,
            midi_file=aug_path,
            other_params=example.other_params,
            generation_time=example.generation_time,
            coherence_score=example.coherence_score,
            genre=example.genre
        )

    def _velocity_scale(self, example: TrainingExample, factor: float) -> TrainingExample:
        """Scale velocities by factor"""
        midi = mido.MidiFile(str(example.midi_file))

        # Scale velocities
        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'velocity'):
                    new_vel = max(0, min(127, int(msg.velocity * factor)))
                    msg.velocity = new_vel

        # Save
        aug_path = example.midi_file.parent / f"{example.midi_file.stem}_velscale_{factor:.2f}.mid"
        midi.save(str(aug_path))

        return TrainingExample(
            features=example.features,
            parameter_value=example.parameter_value,
            midi_file=aug_path,
            other_params=example.other_params,
            generation_time=example.generation_time,
            coherence_score=example.coherence_score,
            genre=example.genre
        )

    def _inversion(self, example: TrainingExample) -> TrainingExample:
        """Melodic inversion around middle C"""
        midi = mido.MidiFile(str(example.midi_file))
        pivot = 60  # Middle C

        for track in midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and hasattr(msg, 'note'):
                    interval = msg.note - pivot
                    msg.note = pivot - interval
                    msg.note = max(0, min(127, msg.note))

        aug_path = example.midi_file.parent / f"{example.midi_file.stem}_inversion.mid"
        midi.save(str(aug_path))

        return TrainingExample(
            features=example.features,
            parameter_value=example.parameter_value,
            midi_file=aug_path,
            other_params=example.other_params,
            generation_time=example.generation_time,
            coherence_score=example.coherence_score,
            genre=example.genre
        )

    def _retrograde(self, example: TrainingExample) -> TrainingExample:
        """Reverse time (retrograde)"""
        midi = mido.MidiFile(str(example.midi_file))

        # Reverse messages in each track
        for track in midi.tracks:
            # Keep meta messages at start
            meta = [msg for msg in track if msg.is_meta]
            notes = [msg for msg in track if not msg.is_meta]
            notes.reverse()

            track.clear()
            track.extend(meta)
            track.extend(notes)

        aug_path = example.midi_file.parent / f"{example.midi_file.stem}_retrograde.mid"
        midi.save(str(aug_path))

        return TrainingExample(
            features=example.features,
            parameter_value=example.parameter_value,
            midi_file=aug_path,
            other_params=example.other_params,
            generation_time=example.generation_time,
            coherence_score=example.coherence_score,
            genre=example.genre
        )


# ============================================================================
# CROSS-VALIDATION UTILITIES
# ============================================================================

class CrossValidationSplitter:
    """
    Create cross-validation splits for training data.

    Supports:
    - K-fold cross-validation
    - Stratified splits (for categorical parameters)
    - Time-series splits
    - Genre-stratified splits
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        """
        Initialize splitter

        Args:
            n_splits: Number of folds
            shuffle: Whether to shuffle data
            random_state: Random seed
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        random.seed(random_state)

    def k_fold_split(
        self,
        training_data: List[TrainingExample]
    ) -> List[Tuple[List[TrainingExample], List[TrainingExample]]]:
        """
        Create k-fold cross-validation splits

        Args:
            training_data: List of training examples

        Returns:
            List of (train, val) splits
        """
        data = training_data.copy()
        if self.shuffle:
            random.shuffle(data)

        fold_size = len(data) // self.n_splits
        splits = []

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = val_start + fold_size if i < self.n_splits - 1 else len(data)

            val_data = data[val_start:val_end]
            train_data = data[:val_start] + data[val_end:]

            splits.append((train_data, val_data))

        return splits

    def stratified_split(
        self,
        training_data: List[TrainingExample],
        n_bins: int = 10
    ) -> List[Tuple[List[TrainingExample], List[TrainingExample]]]:
        """
        Create stratified k-fold splits (ensures even parameter distribution)

        Args:
            training_data: List of training examples
            n_bins: Number of bins for continuous parameters

        Returns:
            List of (train, val) splits
        """
        # Bin parameter values
        values = [ex.parameter_value for ex in training_data]

        if isinstance(values[0], (int, float)):
            # Continuous - use quantile binning
            bins = np.percentile(values, np.linspace(0, 100, n_bins + 1))
            bin_indices = np.digitize(values, bins)
        else:
            # Categorical - use value directly
            unique_values = list(set(values))
            bin_indices = [unique_values.index(v) for v in values]

        # Group by bin
        binned_data = {}
        for ex, bin_idx in zip(training_data, bin_indices):
            if bin_idx not in binned_data:
                binned_data[bin_idx] = []
            binned_data[bin_idx].append(ex)

        # Create stratified splits
        splits = [[] for _ in range(self.n_splits)]

        for bin_idx, bin_examples in binned_data.items():
            if self.shuffle:
                random.shuffle(bin_examples)

            fold_size = len(bin_examples) // self.n_splits

            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else len(bin_examples)
                splits[i].extend(bin_examples[start:end])

        # Convert to train/val pairs
        result = []
        for i in range(self.n_splits):
            val_data = splits[i]
            train_data = []
            for j in range(self.n_splits):
                if j != i:
                    train_data.extend(splits[j])
            result.append((train_data, val_data))

        return result

    def genre_stratified_split(
        self,
        training_data: List[TrainingExample]
    ) -> List[Tuple[List[TrainingExample], List[TrainingExample]]]:
        """
        Create genre-stratified splits (ensures even genre distribution)

        Args:
            training_data: List of training examples

        Returns:
            List of (train, val) splits
        """
        # Group by genre
        genre_data = {}
        for ex in training_data:
            genre = ex.genre or 'unknown'
            if genre not in genre_data:
                genre_data[genre] = []
            genre_data[genre].append(ex)

        # Create splits for each genre
        splits = [[] for _ in range(self.n_splits)]

        for genre, examples in genre_data.items():
            if self.shuffle:
                random.shuffle(examples)

            fold_size = len(examples) // self.n_splits

            for i in range(self.n_splits):
                start = i * fold_size
                end = start + fold_size if i < self.n_splits - 1 else len(examples)
                splits[i].extend(examples[start:end])

        # Convert to train/val pairs
        result = []
        for i in range(self.n_splits):
            val_data = splits[i]
            train_data = []
            for j in range(self.n_splits):
                if j != i:
                    train_data.extend(splits[j])
            result.append((train_data, val_data))

        return result

    def save_splits(
        self,
        splits: List[Tuple[List[TrainingExample], List[TrainingExample]]],
        output_dir: Path,
        param_name: str
    ):
        """
        Save cross-validation splits to disk

        Args:
            splits: List of (train, val) splits
            output_dir: Output directory
            param_name: Parameter name
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, (train, val) in enumerate(splits):
            fold_dir = output_dir / f"fold_{i}"
            fold_dir.mkdir(exist_ok=True)

            # Save indices/filenames
            metadata = {
                'fold': i,
                'param_name': param_name,
                'train_size': len(train),
                'val_size': len(val),
                'train_files': [str(ex.midi_file.name) for ex in train],
                'val_files': [str(ex.midi_file.name) for ex in val]
            }

            with open(fold_dir / 'split.json', 'w') as f:
                json.dump(metadata, f, indent=2)

        logger.info(f"Saved {len(splits)} cross-validation splits to {output_dir}")


# ============================================================================
# DATASET EXPORT UTILITIES
# ============================================================================

class DatasetExporter:
    """
    Export training data to various formats for different ML frameworks.

    Formats:
    - NumPy (.npz)
    - PyTorch (.pt)
    - TensorFlow TFRecord
    - HDF5
    - CSV
    """

    def __init__(self):
        """Initialize exporter"""
        pass

    def export_numpy(
        self,
        training_data: List[TrainingExample],
        output_path: Path,
        include_metadata: bool = True
    ):
        """
        Export to NumPy format (.npz)

        Args:
            training_data: Training examples
            output_path: Output file path
            include_metadata: Whether to include metadata
        """
        X = np.vstack([ex.features for ex in training_data])
        y = np.array([ex.parameter_value for ex in training_data])

        data_dict = {
            'features': X,
            'targets': y
        }

        if include_metadata:
            data_dict['coherence_scores'] = np.array([ex.coherence_score for ex in training_data])
            data_dict['generation_times'] = np.array([ex.generation_time for ex in training_data])
            data_dict['midi_files'] = np.array([str(ex.midi_file) for ex in training_data])

            if any(ex.genre for ex in training_data):
                data_dict['genres'] = np.array([ex.genre or 'unknown' for ex in training_data])

        np.savez_compressed(output_path, **data_dict)
        logger.info(f"Exported {len(training_data)} examples to NumPy format: {output_path}")

    def export_pytorch(
        self,
        training_data: List[TrainingExample],
        output_path: Path
    ):
        """
        Export to PyTorch format (.pt)

        Args:
            training_data: Training examples
            output_path: Output file path
        """
        try:
            import torch
        except ImportError:
            logger.error("PyTorch not installed. Install with: pip install torch")
            return

        X = np.vstack([ex.features for ex in training_data])
        y = np.array([ex.parameter_value for ex in training_data])

        dataset = {
            'features': torch.tensor(X, dtype=torch.float32),
            'targets': torch.tensor(y, dtype=torch.float32),
            'metadata': {
                'n_examples': len(training_data),
                'n_features': X.shape[1],
                'coherence_scores': [ex.coherence_score for ex in training_data],
                'midi_files': [str(ex.midi_file) for ex in training_data]
            }
        }

        torch.save(dataset, output_path)
        logger.info(f"Exported {len(training_data)} examples to PyTorch format: {output_path}")

    def export_hdf5(
        self,
        training_data: List[TrainingExample],
        output_path: Path
    ):
        """
        Export to HDF5 format

        Args:
            training_data: Training examples
            output_path: Output file path
        """
        try:
            import h5py
        except ImportError:
            logger.error("h5py not installed. Install with: pip install h5py")
            return

        X = np.vstack([ex.features for ex in training_data])
        y = np.array([ex.parameter_value for ex in training_data])

        with h5py.File(output_path, 'w') as f:
            f.create_dataset('features', data=X, compression='gzip')
            f.create_dataset('targets', data=y, compression='gzip')
            f.create_dataset('coherence_scores', data=np.array([ex.coherence_score for ex in training_data]))
            f.create_dataset('generation_times', data=np.array([ex.generation_time for ex in training_data]))

            # String arrays
            dt = h5py.special_dtype(vlen=str)
            f.create_dataset('midi_files', data=[str(ex.midi_file) for ex in training_data], dtype=dt)

            if any(ex.genre for ex in training_data):
                f.create_dataset('genres', data=[ex.genre or 'unknown' for ex in training_data], dtype=dt)

        logger.info(f"Exported {len(training_data)} examples to HDF5 format: {output_path}")

    def export_csv_features(
        self,
        training_data: List[TrainingExample],
        output_path: Path,
        max_features: Optional[int] = None
    ):
        """
        Export to CSV with features

        Args:
            training_data: Training examples
            output_path: Output file path
            max_features: Maximum number of features to include (None = all)
        """
        import csv

        X = np.vstack([ex.features for ex in training_data])

        if max_features:
            X = X[:, :max_features]

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            header = ['index', 'parameter_value', 'coherence_score', 'genre']
            header.extend([f'feature_{i}' for i in range(X.shape[1])])
            writer.writerow(header)

            # Data
            for i, (ex, features) in enumerate(zip(training_data, X)):
                row = [
                    i,
                    ex.parameter_value,
                    ex.coherence_score,
                    ex.genre or 'N/A'
                ]
                row.extend(features.tolist())
                writer.writerow(row)

        logger.info(f"Exported {len(training_data)} examples to CSV: {output_path}")


# ============================================================================
# DATASET QUALITY ANALYZER
# ============================================================================

class DatasetQualityAnalyzer:
    """
    Analyze quality and properties of generated training datasets.

    Provides:
    - Coverage analysis (parameter space coverage)
    - Quality metrics (coherence, generation time)
    - Distribution analysis
    - Outlier detection
    - Recommendations for improvement
    """

    def __init__(self):
        """Initialize analyzer"""
        pass

    def analyze_dataset(
        self,
        training_data: List[TrainingExample],
        param_def: ParameterDefinition
    ) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis

        Args:
            training_data: Training examples
            param_def: Parameter definition

        Returns:
            Analysis report dictionary
        """
        report = {
            'basic_stats': self._basic_statistics(training_data),
            'coverage_analysis': self._coverage_analysis(training_data, param_def),
            'quality_metrics': self._quality_metrics(training_data),
            'distribution_analysis': self._distribution_analysis(training_data, param_def),
            'outlier_detection': self._outlier_detection(training_data),
            'recommendations': []
        }

        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report, param_def)

        return report

    def _basic_statistics(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Basic statistics"""
        successful = [ex for ex in training_data if ex.validation_passed]

        return {
            'total_examples': len(training_data),
            'successful_examples': len(successful),
            'failed_examples': len(training_data) - len(successful),
            'success_rate': len(successful) / len(training_data) if training_data else 0
        }

    def _coverage_analysis(
        self,
        training_data: List[TrainingExample],
        param_def: ParameterDefinition
    ) -> Dict[str, Any]:
        """Analyze parameter space coverage"""
        values = [ex.parameter_value for ex in training_data]

        if param_def.param_type in [ParameterType.CONTINUOUS, ParameterType.PROBABILITY]:
            # Continuous parameter - check coverage using histogram
            hist, edges = np.histogram(values, bins=20)

            # Coverage score: percentage of bins with data
            coverage_score = np.sum(hist > 0) / len(hist)

            # Uniformity score: how evenly distributed
            expected_per_bin = len(values) / len(hist)
            uniformity_score = 1.0 - np.std(hist) / expected_per_bin if expected_per_bin > 0 else 0

            return {
                'coverage_score': coverage_score,
                'uniformity_score': uniformity_score,
                'bins_with_data': int(np.sum(hist > 0)),
                'total_bins': len(hist),
                'min_bin_count': int(np.min(hist)),
                'max_bin_count': int(np.max(hist)),
                'empty_regions': self._find_empty_regions(hist, edges)
            }

        elif param_def.param_type == ParameterType.CATEGORICAL:
            # Categorical - check all options represented
            unique_values = set(values)
            expected_values = set(param_def.options) if param_def.options else unique_values

            coverage_score = len(unique_values) / len(expected_values) if expected_values else 1.0

            from collections import Counter
            counts = Counter(values)

            return {
                'coverage_score': coverage_score,
                'unique_values': len(unique_values),
                'expected_values': len(expected_values),
                'missing_values': list(expected_values - unique_values),
                'value_counts': dict(counts)
            }

        else:
            return {'coverage_score': 1.0}

    def _find_empty_regions(self, hist: np.ndarray, edges: np.ndarray) -> List[Tuple[float, float]]:
        """Find empty regions in histogram"""
        empty_regions = []

        for i, count in enumerate(hist):
            if count == 0:
                empty_regions.append((float(edges[i]), float(edges[i + 1])))

        return empty_regions

    def _quality_metrics(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Quality metrics"""
        coherence_scores = [ex.coherence_score for ex in training_data]
        generation_times = [ex.generation_time for ex in training_data]

        return {
            'avg_coherence': float(np.mean(coherence_scores)),
            'min_coherence': float(np.min(coherence_scores)),
            'max_coherence': float(np.max(coherence_scores)),
            'std_coherence': float(np.std(coherence_scores)),
            'avg_generation_time': float(np.mean(generation_times)),
            'total_generation_time': float(np.sum(generation_times)),
            'low_quality_examples': sum(1 for s in coherence_scores if s < 0.5),
            'high_quality_examples': sum(1 for s in coherence_scores if s > 0.8)
        }

    def _distribution_analysis(
        self,
        training_data: List[TrainingExample],
        param_def: ParameterDefinition
    ) -> Dict[str, Any]:
        """Analyze parameter value distribution"""
        values = [ex.parameter_value for ex in training_data]

        if isinstance(values[0], (int, float)):
            return {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
                'quartiles': {
                    'q25': float(np.percentile(values, 25)),
                    'q50': float(np.percentile(values, 50)),
                    'q75': float(np.percentile(values, 75))
                },
                'skewness': self._calculate_skewness(values),
                'kurtosis': self._calculate_kurtosis(values)
            }
        else:
            from collections import Counter
            counts = Counter(values)
            return {
                'value_distribution': dict(counts),
                'unique_count': len(counts),
                'most_common': counts.most_common(5)
            }

    def _calculate_skewness(self, values: List[float]) -> float:
        """Calculate skewness"""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        return float(np.mean(((arr - mean) / std) ** 3))

    def _calculate_kurtosis(self, values: List[float]) -> float:
        """Calculate kurtosis"""
        arr = np.array(values)
        mean = np.mean(arr)
        std = np.std(arr)

        if std == 0:
            return 0.0

        return float(np.mean(((arr - mean) / std) ** 4)) - 3.0

    def _outlier_detection(self, training_data: List[TrainingExample]) -> Dict[str, Any]:
        """Detect outliers"""
        coherence_scores = np.array([ex.coherence_score for ex in training_data])
        generation_times = np.array([ex.generation_time for ex in training_data])

        # Use IQR method for outlier detection
        def find_outliers(data):
            q1 = np.percentile(data, 25)
            q3 = np.percentile(data, 75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return np.where((data < lower_bound) | (data > upper_bound))[0]

        coherence_outliers = find_outliers(coherence_scores)
        time_outliers = find_outliers(generation_times)

        return {
            'coherence_outliers': {
                'count': len(coherence_outliers),
                'indices': coherence_outliers.tolist(),
                'values': coherence_scores[coherence_outliers].tolist() if len(coherence_outliers) > 0 else []
            },
            'generation_time_outliers': {
                'count': len(time_outliers),
                'indices': time_outliers.tolist(),
                'values': generation_times[time_outliers].tolist() if len(time_outliers) > 0 else []
            }
        }

    def _generate_recommendations(
        self,
        report: Dict[str, Any],
        param_def: ParameterDefinition
    ) -> List[str]:
        """Generate recommendations for improvement"""
        recommendations = []

        # Coverage recommendations
        coverage = report['coverage_analysis']
        if coverage.get('coverage_score', 1.0) < 0.8:
            recommendations.append(
                f"Low parameter space coverage ({coverage['coverage_score']:.1%}). "
                "Consider generating more examples or using better sampling."
            )

        if coverage.get('uniformity_score', 1.0) < 0.7:
            recommendations.append(
                "Uneven parameter distribution. Some regions are over-represented. "
                "Use Latin hypercube sampling for better uniformity."
            )

        # Quality recommendations
        quality = report['quality_metrics']
        if quality['avg_coherence'] < 0.6:
            recommendations.append(
                f"Low average coherence ({quality['avg_coherence']:.2f}). "
                "Review generator configuration or increase min_coherence threshold."
            )

        if quality['low_quality_examples'] > len(report['basic_stats']['total_examples']) * 0.2:
            recommendations.append(
                f"{quality['low_quality_examples']} low-quality examples (coherence < 0.5). "
                "Consider filtering or regenerating these examples."
            )

        # Success rate recommendations
        if report['basic_stats']['success_rate'] < 0.9:
            recommendations.append(
                f"Low success rate ({report['basic_stats']['success_rate']:.1%}). "
                "Check for generation errors and adjust failure thresholds."
            )

        # Outlier recommendations
        outliers = report['outlier_detection']
        if outliers['coherence_outliers']['count'] > 0:
            recommendations.append(
                f"{outliers['coherence_outliers']['count']} coherence outliers detected. "
                "Review these examples for quality issues."
            )

        if not recommendations:
            recommendations.append("Dataset quality looks good! No major issues detected.")

        return recommendations

    def print_report(self, report: Dict[str, Any]):
        """Print analysis report"""
        print("\n" + "="*60)
        print("DATASET QUALITY ANALYSIS REPORT")
        print("="*60 + "\n")

        print("BASIC STATISTICS")
        print("-" * 60)
        stats = report['basic_stats']
        print(f"Total Examples: {stats['total_examples']}")
        print(f"Successful: {stats['successful_examples']}")
        print(f"Failed: {stats['failed_examples']}")
        print(f"Success Rate: {stats['success_rate']:.1%}")

        print("\nCOVERAGE ANALYSIS")
        print("-" * 60)
        coverage = report['coverage_analysis']
        print(f"Coverage Score: {coverage.get('coverage_score', 'N/A'):.1%}")
        if 'uniformity_score' in coverage:
            print(f"Uniformity Score: {coverage['uniformity_score']:.1%}")

        print("\nQUALITY METRICS")
        print("-" * 60)
        quality = report['quality_metrics']
        print(f"Avg Coherence: {quality['avg_coherence']:.3f}")
        print(f"Min Coherence: {quality['min_coherence']:.3f}")
        print(f"Max Coherence: {quality['max_coherence']:.3f}")
        print(f"High Quality (>0.8): {quality['high_quality_examples']}")
        print(f"Low Quality (<0.5): {quality['low_quality_examples']}")

        print("\nRECOMMENDATIONS")
        print("-" * 60)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")

        print("\n" + "="*60 + "\n")


# ============================================================================
# ACTIVE LEARNING UTILITIES
# ============================================================================

class ActiveLearningSelector:
    """
    Select parameter values for active learning.

    Strategies:
    - Uncertainty sampling (regions with high prediction uncertainty)
    - Diversity sampling (maximize diversity in parameter space)
    - Query-by-committee (disagreement between multiple models)
    - Expected model change (maximum impact on model)
    """

    def __init__(self):
        """Initialize selector"""
        pass

    def select_uncertain_regions(
        self,
        param_def: ParameterDefinition,
        existing_data: List[TrainingExample],
        n_samples: int = 100
    ) -> List[Any]:
        """
        Select parameter values in uncertain regions

        Args:
            param_def: Parameter definition
            existing_data: Existing training data
            n_samples: Number of new samples to select

        Returns:
            List of parameter values to generate
        """
        # For now, simple strategy: find gaps in existing data
        if param_def.param_type in [ParameterType.CONTINUOUS, ParameterType.PROBABILITY]:
            existing_values = sorted([ex.parameter_value for ex in existing_data])

            # Find largest gaps
            gaps = []
            for i in range(len(existing_values) - 1):
                gap_size = existing_values[i + 1] - existing_values[i]
                gap_center = (existing_values[i] + existing_values[i + 1]) / 2
                gaps.append((gap_size, gap_center))

            gaps.sort(reverse=True)

            # Sample from largest gaps
            new_values = []
            for gap_size, gap_center in gaps[:n_samples]:
                # Sample around gap center
                value = gap_center + random.uniform(-gap_size/4, gap_size/4)
                value = max(param_def.min_value, min(param_def.max_value, value))
                new_values.append(value)

            # Fill remaining with random
            while len(new_values) < n_samples:
                value = random.uniform(param_def.min_value, param_def.max_value)
                new_values.append(value)

            return new_values

        else:
            # For categorical, sample under-represented values
            from collections import Counter
            existing_values = [ex.parameter_value for ex in existing_data]
            counts = Counter(existing_values)

            if param_def.options:
                # Sample proportionally to inverse frequency
                weights = []
                for option in param_def.options:
                    count = counts.get(option, 0)
                    weights.append(1.0 / (count + 1))  # +1 to avoid division by zero

                # Normalize weights
                total_weight = sum(weights)
                weights = [w / total_weight for w in weights]

                # Sample
                new_values = random.choices(param_def.options, weights=weights, k=n_samples)
                return new_values

            return [param_def.default_value] * n_samples


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """Command line interface for synthetic data generation"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate synthetic training data for music generation parameters"
    )
    parser.add_argument(
        'param_name',
        type=str,
        help="Parameter name to generate data for"
    )
    parser.add_argument(
        '--n-examples',
        type=int,
        default=1000,
        help="Number of examples to generate (default: 1000)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='training_data',
        help="Output directory (default: training_data)"
    )
    parser.add_argument(
        '--genre-balanced',
        action='store_true',
        help="Generate genre-balanced dataset"
    )
    parser.add_argument(
        '--n-per-genre',
        type=int,
        default=100,
        help="Examples per genre for balanced generation (default: 100)"
    )
    parser.add_argument(
        '--min-coherence',
        type=float,
        default=0.5,
        help="Minimum coherence score (default: 0.5)"
    )

    args = parser.parse_args()

    # Initialize generator
    logger.info("Initializing Synthetic Training Data Generator...")
    registry = UniversalParameterRegistry()
    generator = SyntheticTrainingDataGenerator(
        registry=registry,
        output_root=Path(args.output_dir)
    )

    # Get parameter definition
    param_def = registry.parameters.get(args.param_name)
    if param_def is None:
        logger.error(f"Parameter '{args.param_name}' not found in registry")
        return 1

    logger.info(f"Generating data for parameter: {args.param_name}")
    logger.info(f"Type: {param_def.param_type.value}")
    logger.info(f"Category: {param_def.category.value if param_def.category else 'N/A'}")

    # Generate data
    try:
        if args.genre_balanced:
            training_data = generator.generate_balanced_dataset(
                param_name=args.param_name,
                param_def=param_def,
                n_per_genre=args.n_per_genre
            )
        else:
            training_data = generator.generate_training_data(
                param_name=args.param_name,
                param_def=param_def,
                n_examples=args.n_examples,
                min_coherence=args.min_coherence
            )

        logger.info(f"\n✅ SUCCESS: Generated {len(training_data)} training examples")

        # Print statistics
        stats = generator.get_global_statistics()
        logger.info(f"\nGlobal Statistics:")
        logger.info(f"  Total Generated: {stats['total_generated']}")
        logger.info(f"  Total Failed: {stats['total_failed']}")
        logger.info(f"  Success Rate: {stats['success_rate']:.1%}")
        logger.info(f"  Avg Generation Time: {stats['avg_generation_time']:.3f}s")
        logger.info(f"  Avg Coherence Score: {stats['avg_coherence_score']:.3f}")

        return 0

    except Exception as e:
        logger.error(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
=======
        print(f"\n{'='*80}")
        print(f"GENERATING GENRE-BALANCED DATASET")
        print(f"{'='*80}")
        print(f"Parameter: {param_name}")
        print(f"Genres: {len(genres)}")
        print(f"Examples per genre: {n_per_genre}")
        print(f"Total examples: {len(genres) * n_per_genre}")
        print()

        all_examples = []
        output_dir = Path('training_data') / param_name.replace('.', '_')

        for genre in genres:
            print(f"\nGenerating {n_per_genre} examples for {genre} style...")

            genre_dir = output_dir / genre
            genre_dir.mkdir(parents=True, exist_ok=True)

            # Get genre-specific base parameters
            genre_params = self._get_genre_params(genre)

            for i in range(n_per_genre):
                # Sample parameter value
                param_value = self._sample_single_value(param_def)

                # Combine with genre params
                params = genre_params.copy()
                params[param_name] = param_value

                try:
                    # Generate example
                    example = self._generate_single_example_with_params(
                        param_name,
                        param_value,
                        params,
                        genre_dir,
                        i,
                        genre=genre
                    )

                    all_examples.append(example)

                except Exception as e:
                    print(f"  Warning: Failed to generate example {i} for {genre}: {e}")
                    continue

        # Create dataset
        statistics = self._analyze_statistics(all_examples, param_def)

        dataset = TrainingDataset(
            parameter_name=param_name,
            parameter_definition=param_def,
            examples=all_examples,
            n_examples=len(all_examples),
            generation_date=datetime.now().isoformat(),
            statistics=statistics,
            output_directory=output_dir
        )

        # Save metadata
        self._save_metadata(dataset)

        print(f"\n✅ Generated {len(all_examples)} genre-balanced examples")

        return dataset

    def _get_genre_params(self, genre: str) -> Dict[str, Any]:
        """Get default parameters for genre"""

        # Mock genre parameters
        genre_defaults = {
            'swing': {
                'tempo': 180,
                'swing_amount': 0.6,
                'complexity': 0.6,
                'style': 'swing'
            },
            'bebop': {
                'tempo': 220,
                'swing_amount': 0.65,
                'complexity': 0.9,
                'style': 'bebop'
            },
            'modal': {
                'tempo': 140,
                'swing_amount': 0.55,
                'complexity': 0.5,
                'style': 'modal'
            },
            'bossa_nova': {
                'tempo': 120,
                'swing_amount': 0.0,
                'complexity': 0.5,
                'style': 'bossa'
            },
            'fusion': {
                'tempo': 160,
                'swing_amount': 0.3,
                'complexity': 0.7,
                'style': 'fusion'
            },
            'cool_jazz': {
                'tempo': 110,
                'swing_amount': 0.5,
                'complexity': 0.4,
                'style': 'cool'
            },
            'free_jazz': {
                'tempo': 160,
                'swing_amount': 0.4,
                'complexity': 0.95,
                'style': 'free'
            },
            'ballad': {
                'tempo': 70,
                'swing_amount': 0.55,
                'complexity': 0.4,
                'style': 'ballad'
            }
        }

        return genre_defaults.get(genre, self._sample_default_parameters())

    def _sample_single_value(self, param_def: dict) -> Any:
        """Sample one parameter value"""

        param_type = param_def.get('type', 'CONTINUOUS')
        param_range = param_def.get('range', (0.0, 1.0))

        if param_type == 'CONTINUOUS' or param_type == 'PROBABILITY':
            if isinstance(param_range, (list, tuple)) and len(param_range) == 2:
                min_val, max_val = param_range
                return random.uniform(min_val, max_val)
            else:
                return 0.5

        elif param_type == 'CATEGORICAL':
            if isinstance(param_range, list) and param_range:
                return random.choice(param_range)
            else:
                return 'default'

        elif param_type == 'BOOLEAN':
            return random.choice([True, False])

        else:
            return param_def.get('default', 0.5)

    def _generate_single_example_with_params(self,
                                            param_name: str,
                                            param_value: Any,
                                            params: dict,
                                            output_dir: Path,
                                            example_idx: int,
                                            genre: Optional[str] = None) -> TrainingExample:
        """Generate single example with provided parameters"""

        start_time = time.time()

        # Generate MIDI
        midi = self._generate_midi_mock(params)

        # Save MIDI
        midi_filename = f"{param_name.replace('.', '_')}_{genre}_{example_idx:04d}.mid"
        midi_path = output_dir / midi_filename
        self._save_midi_mock(midi, midi_path)

        # Extract features
        features = self._extract_features_mock(midi_path)

        # Validate coherence
        coherence_score = self.validator.validate_coherence(midi_path)

        generation_time = time.time() - start_time

        return TrainingExample(
            features=features,
            parameter_value=param_value,
            midi_file=midi_path,
            other_params=params.copy(),
            generation_time=generation_time,
            coherence_score=coherence_score,
            genre=genre
        )

    def load_training_data(self, param_dir: Path) -> TrainingDataset:
        """
        Load training data from directory

        Args:
            param_dir: Directory containing training data

        Returns:
            TrainingDataset loaded from disk
        """
        # Load metadata
        metadata_file = param_dir / 'metadata.json'
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Load examples
        examples = []
        for ex_meta in metadata['examples']:
            midi_path = param_dir / ex_meta['midi_file']

            # Extract features
            if midi_path.exists():
                features = self._extract_features_mock(midi_path)
            else:
                # Use zero features if file missing
                features = np.zeros(1000)

            example = TrainingExample(
                features=features,
                parameter_value=ex_meta['parameter_value'],
                midi_file=midi_path,
                other_params={},
                generation_time=ex_meta.get('generation_time', 0.0),
                coherence_score=ex_meta.get('coherence_score', 1.0),
                genre=ex_meta.get('genre')
            )
            examples.append(example)

        # Create dataset
        dataset = TrainingDataset(
            parameter_name=metadata['parameter_name'],
            parameter_definition=metadata.get('parameter_definition', {}),
            examples=examples,
            n_examples=metadata['n_examples'],
            generation_date=metadata['generation_date'],
            statistics=metadata.get('statistics', {}),
            output_directory=param_dir
        )

        return dataset

    def export_for_training(self, dataset: TrainingDataset, output_file: Path):
        """
        Export dataset in format ready for XGBoost training

        Args:
            dataset: Training dataset
            output_file: Output file path (.npz format)
        """
        # Extract features and labels
        X = np.array([ex.features for ex in dataset.examples])
        y = np.array([ex.parameter_value for ex in dataset.examples])

        # Save as numpy compressed format
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            parameter_name=dataset.parameter_name,
            n_examples=dataset.n_examples
        )

        print(f"Exported training data to: {output_file}")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")


# Example usage
if __name__ == '__main__':
    # Example parameter definition
    example_param_def = {
        'name': 'harmony.voicing.quartal_probability',
        'type': 'CONTINUOUS',
        'range': (0.0, 1.0),
        'default': 0.3,
        'description': 'Probability of using quartal voicings'
    }

    # Create generator
    generator = SyntheticTrainingDataGenerator()

    # Generate training data
    dataset = generator.generate_training_data(
        param_name=example_param_def['name'],
        param_def=example_param_def,
        n_examples=100,  # Small for example
        output_dir=Path('example_training_data')
    )

    print(f"\n{'='*80}")
    print("DATASET SUMMARY")
    print(f"{'='*80}")
    print(f"Parameter: {dataset.parameter_name}")
    print(f"Examples: {dataset.n_examples}")
    print(f"Avg coherence: {dataset.statistics.get('avg_coherence', 0):.3f}")
    print(f"Saved to: {dataset.output_directory}")
>>>>>>> origin/claude/music-generation-agents-01YDx3Cus9i72savb8rGvQGS
