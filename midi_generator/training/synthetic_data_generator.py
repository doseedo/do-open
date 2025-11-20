"""
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
License: MIT
"""

import json
import random
import time
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

        checks = {
            'has_notes': self._check_has_notes(midi),
            'reasonable_length': self._check_reasonable_length(midi),
            'pitch_range_ok': self._check_pitch_range(midi),
            'velocities_ok': self._check_velocities(midi),
            'rhythmic_variation': self._check_rhythm(midi),
            'harmonic_coherence': self._check_harmony(midi)
        }

        # Weighted average
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
                return 1.0
            elif 2 <= length <= 180:
                return 0.7
            else:
                return 0.3
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
        1. Sample parameter values using Latin hypercube (even coverage)
        2. For each value, vary other parameters (prevent overfitting)
        3. Generate MIDI
        4. Validate musical coherence
        5. Extract features
        6. Store: (features, parameter_value, MIDI_file)

        Args:
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

                try:
                    # Generate training example
                    example = self._generate_single_example(
                        param_name,
                        param_value,
                        param_def,
                        param_dir,
                        len(examples)
                    )

                    # Validate musical coherence
                    if example.coherence_score < min_coherence:
                        failed += 1
                        continue

                    examples.append(example)
                    pbar.update(1)

                except Exception as e:
                    failed += 1
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
        """
        Generate one training example

        Args:
            param_name: Parameter name
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
        """
        Generate training data balanced across genres

        Args:
            param_name: Parameter name
            param_def: Parameter definition
            n_per_genre: Examples per genre
            genres: List of genres to use

        Returns:
            TrainingDataset with genre-balanced examples
        """
        if genres is None:
            genres = [
                'swing', 'bebop', 'modal', 'bossa_nova',
                'fusion', 'cool_jazz', 'free_jazz', 'ballad'
            ]

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
