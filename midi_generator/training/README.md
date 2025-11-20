# Synthetic Training Data Generator - Agent 14

## Overview

The Synthetic Training Data Generator is a comprehensive system for creating high-quality training datasets for new parameters in the Musical Program Synthesis system. It generates diverse, musically coherent MIDI examples with extracted features for training XGBoost models.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│          Synthetic Training Data Generator                  │
│                      (Agent 14)                             │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Parameter  │  │   Musical    │  │   Feature    │
│   Sampler    │  │  Coherence   │  │  Extractor   │
│   (Latin     │  │  Validator   │  │              │
│  Hypercube)  │  │              │  │              │
└──────────────┘  └──────────────┘  └──────────────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
                    ┌──────────────┐
                    │   Training   │
                    │   Examples   │
                    │  (1000+ per  │
                    │  parameter)  │
                    └──────────────┘
```

## Key Features

### 1. **Latin Hypercube Sampling**
- Even coverage across parameter space
- No clustering or gaps in sampled values
- Optimal for training robust ML models

### 2. **Musical Coherence Validation**
Validates 6 aspects of musical quality:
- **Note presence**: Ensures MIDI has actual notes
- **Length**: Reasonable duration (5-120 seconds)
- **Pitch range**: Valid MIDI pitches (21-108)
- **Velocities**: Appropriate dynamics (30-100 average)
- **Rhythm**: Coherent timing patterns
- **Harmony**: Basic harmonic validity

### 3. **Genre-Balanced Generation**
Generates examples across multiple genres:
- Swing
- Bebop
- Modal
- Bossa Nova
- Fusion
- Cool Jazz
- Free Jazz
- Ballad

### 4. **Diverse Parameter Variation**
When training parameter X:
- All other parameters are randomly varied
- Prevents overfitting to specific parameter combinations
- Creates robust, generalizable models

### 5. **Comprehensive Metadata**
Tracks and saves:
- Parameter value distributions
- Coherence score statistics
- Generation times
- Genre distributions
- Quality metrics
- Failed examples (for debugging)

### 6. **Robust Error Handling**
- Automatic retry on generation failures
- Configurable failure thresholds
- Detailed error logging
- Graceful degradation

## Installation

### Dependencies

```bash
# Core dependencies
pip install numpy scipy mido tqdm

# Optional (for full functionality)
pip install pandas matplotlib
```

### Required System Components

The generator integrates with:
- `UniversalParameterRegistry` - Parameter definitions
- `HarmonyModule` - MIDI generation (or any generator)
- `DeepFeatureExtractor` - Feature extraction
- `StyleRegistry` - Genre-specific parameters

## Usage

### Basic Usage

```python
from midi_generator.training import SyntheticTrainingDataGenerator
from midi_generator.parameters.universal_registry import UniversalParameterRegistry

# Initialize
registry = UniversalParameterRegistry()
generator = SyntheticTrainingDataGenerator()

# Get parameter definition
param_name = "harmony.jazz.voicing_density"
param_def = registry.parameters[param_name]

# Generate 1000 training examples
training_data = generator.generate_training_data(
    param_name=param_name,
    param_def=param_def,
    n_examples=1000,
    min_coherence=0.5  # Minimum quality threshold
)

print(f"Generated {len(training_data)} examples")
```

### Genre-Balanced Generation

```python
# Generate 100 examples per genre (800 total)
balanced_data = generator.generate_balanced_dataset(
    param_name=param_name,
    param_def=param_def,
    n_per_genre=100
)

# Check genre distribution
from collections import Counter
genres = Counter([ex.genre for ex in balanced_data])
print(f"Genre distribution: {genres}")
```

### Batch Generation for Multiple Parameters

```python
from midi_generator.training import BatchTrainingDataGenerator

# Initialize batch generator
batch_gen = BatchTrainingDataGenerator()

# Generate for multiple parameters
param_names = [
    "harmony.jazz.voicing_density",
    "harmony.jazz.voicing_spread",
    "melody.bebop.chromaticism"
]

results = batch_gen.generate_for_multiple_parameters(
    param_names=param_names,
    n_examples_per_param=1000
)

# Results is dict: {param_name: [TrainingExample, ...]}
for param, examples in results.items():
    print(f"{param}: {len(examples)} examples")
```

### Custom Validation Criteria

```python
from midi_generator.training import MusicalCoherenceValidator

# Create strict validator
strict_validator = MusicalCoherenceValidator(strict_mode=True)

# Use in generator
generator = SyntheticTrainingDataGenerator(
    validator=strict_validator
)

# Generate with higher quality threshold
training_data = generator.generate_training_data(
    param_name=param_name,
    param_def=param_def,
    n_examples=1000,
    min_coherence=0.7  # Higher threshold
)
```

### Analyzing Results

```python
# Get detailed validation scores
validator = MusicalCoherenceValidator()
midi_file = mido.MidiFile('path/to/generated.mid')

score, details = validator.validate_with_details(midi_file)

print(f"Overall coherence: {score:.2f}")
print(f"Breakdown:")
for aspect, value in details.items():
    print(f"  {aspect}: {value:.2f}")
```

## Command Line Interface

```bash
# Generate data for a parameter
python -m midi_generator.training.synthetic_data_generator \
    harmony.jazz.voicing_density \
    --n-examples 1000 \
    --output-dir training_data \
    --min-coherence 0.5

# Generate genre-balanced dataset
python -m midi_generator.training.synthetic_data_generator \
    harmony.jazz.voicing_density \
    --genre-balanced \
    --n-per-genre 100 \
    --output-dir training_data
```

## Output Structure

```
training_data/
├── harmony_jazz_voicing_density/
│   ├── harmony_jazz_voicing_density_0000.mid
│   ├── harmony_jazz_voicing_density_0001.mid
│   ├── ...
│   ├── harmony_jazz_voicing_density_0999.mid
│   ├── metadata.json
│   └── summary.csv
└── swing/
    └── harmony_jazz_voicing_density/
        ├── harmony_jazz_voicing_density_0000.mid
        ├── ...
        └── metadata.json
```

### Metadata Format

```json
{
  "parameter_name": "harmony.jazz.voicing_density",
  "parameter_type": "continuous",
  "parameter_category": "harmony",
  "generation_date": "2025-11-20T12:34:56",
  "statistics": {
    "total_examples": 1000,
    "successful_examples": 987,
    "failed_examples": 13,
    "avg_coherence_score": 0.85,
    "avg_generation_time": 0.234,
    "parameter_value_distribution": {
      "mean": 0.5,
      "std": 0.29,
      "min": 0.01,
      "max": 0.99,
      "median": 0.51,
      "quartiles": {
        "q25": 0.25,
        "q50": 0.51,
        "q75": 0.75
      }
    },
    "quality_metrics": {
      "coherence_min": 0.50,
      "coherence_max": 0.98,
      "coherence_std": 0.12,
      "coherence_median": 0.87
    }
  },
  "examples": [
    {
      "parameter_value": 0.73,
      "coherence_score": 0.87,
      "generation_time": 0.234,
      "midi_file": "harmony_jazz_voicing_density_0001.mid"
    }
  ]
}
```

## API Reference

### SyntheticTrainingDataGenerator

```python
class SyntheticTrainingDataGenerator:
    def __init__(
        self,
        generator=None,                # MIDI generator
        feature_extractor=None,        # Feature extractor
        registry=None,                 # Parameter registry
        validator=None,                # Coherence validator
        output_root=Path('training_data')
    )

    def generate_training_data(
        self,
        param_name: str,
        param_def: ParameterDefinition,
        n_examples: int = 1000,
        output_dir: Optional[Path] = None,
        min_coherence: float = 0.5,
        max_failures_ratio: float = 0.2
    ) -> List[TrainingExample]

    def generate_balanced_dataset(
        self,
        param_name: str,
        param_def: ParameterDefinition,
        n_per_genre: int = 100,
        genres: Optional[List[str]] = None
    ) -> List[TrainingExample]

    def get_global_statistics(self) -> Dict[str, Any]
```

### MusicalCoherenceValidator

```python
class MusicalCoherenceValidator:
    def __init__(self, strict_mode: bool = False)

    def validate_coherence(self, midi: mido.MidiFile) -> float
    # Returns score 0.0-1.0

    def validate_with_details(self, midi: mido.MidiFile) -> Tuple[float, Dict]
    # Returns (score, {aspect: score})
```

### ParameterSpaceSampler

```python
class ParameterSpaceSampler:
    def __init__(self, seed: Optional[int] = 42)

    def sample_parameter_space(
        self,
        param_def: ParameterDefinition,
        n_samples: int
    ) -> List[Any]
    # Returns evenly distributed samples using Latin hypercube
```

### TrainingExample

```python
@dataclass
class TrainingExample:
    features: np.ndarray           # Extracted features (1000+)
    parameter_value: Any           # Parameter value for this example
    midi_file: Path                # Path to generated MIDI
    other_params: Dict[str, Any]   # Other parameter values
    generation_time: float         # Time to generate (seconds)
    coherence_score: float         # Musical coherence (0.0-1.0)
    genre: Optional[str]           # Genre (if genre-balanced)
    validation_passed: bool        # Whether validation passed
    error_message: Optional[str]   # Error if failed
```

## Performance

### Generation Speed

Typical performance on modern hardware:
- **Generation**: ~0.2-0.5 seconds per example
- **1000 examples**: ~5-10 minutes
- **Genre-balanced (800 examples)**: ~8-15 minutes

### Quality Metrics

Expected quality for well-configured generators:
- **Success rate**: >95%
- **Avg coherence**: >0.80
- **Parameter coverage**: Even distribution across range

## Best Practices

### 1. Start with Small Batches

```python
# Test with 100 examples first
test_data = generator.generate_training_data(
    param_name=param_name,
    param_def=param_def,
    n_examples=100
)

# Check quality
avg_coherence = np.mean([ex.coherence_score for ex in test_data])
if avg_coherence > 0.7:
    # Quality is good, scale up
    full_data = generator.generate_training_data(
        param_name=param_name,
        param_def=param_def,
        n_examples=1000
    )
```

### 2. Monitor Failure Rates

```python
stats = generator.get_global_statistics()
print(f"Success rate: {stats['success_rate']:.1%}")

if stats['success_rate'] < 0.9:
    print("WARNING: High failure rate. Check generator configuration.")
```

### 3. Use Genre-Balanced for Style-Dependent Parameters

```python
# For parameters that affect style significantly
if param_def.category == ParameterCategory.STYLE:
    data = generator.generate_balanced_dataset(
        param_name=param_name,
        param_def=param_def,
        n_per_genre=100
    )
else:
    # Standard generation for other parameters
    data = generator.generate_training_data(
        param_name=param_name,
        param_def=param_def,
        n_examples=1000
    )
```

### 4. Validate Parameter Definitions

```python
# Ensure parameter definition is complete
param_def = registry.parameters[param_name]

assert param_def.param_type is not None
assert param_def.default_value is not None

if param_def.param_type == ParameterType.CONTINUOUS:
    assert param_def.min_value is not None
    assert param_def.max_value is not None
```

## Integration with XGBoost Training

```python
from midi_generator.training import SyntheticTrainingDataGenerator
import xgboost as xgb

# 1. Generate training data
generator = SyntheticTrainingDataGenerator()
training_data = generator.generate_training_data(
    param_name="harmony.jazz.voicing_density",
    param_def=param_def,
    n_examples=1000
)

# 2. Prepare XGBoost dataset
X = np.vstack([ex.features for ex in training_data])
y = np.array([ex.parameter_value for ex in training_data])

# 3. Train XGBoost model
dtrain = xgb.DMatrix(X, label=y)
params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'eta': 0.1
}
model = xgb.train(params, dtrain, num_boost_round=100)

# 4. Save model
model.save_model('voicing_density_model.json')
```

## Troubleshooting

### High Failure Rate

**Problem**: >20% of generations fail

**Solutions**:
1. Check generator configuration
2. Lower `min_coherence` threshold
3. Increase `max_failures_ratio`
4. Verify parameter ranges are valid

### Low Coherence Scores

**Problem**: Avg coherence <0.6

**Solutions**:
1. Review generator output manually
2. Adjust validation weights in `MusicalCoherenceValidator`
3. Check if MIDI files have actual notes
4. Verify tempo and timing are reasonable

### Slow Generation

**Problem**: >1 second per example

**Solutions**:
1. Profile generator code
2. Reduce MIDI length if too long
3. Optimize feature extraction
4. Use batch generation for multiple parameters

### Memory Issues

**Problem**: Out of memory with large datasets

**Solutions**:
1. Generate in smaller batches
2. Don't store all examples in memory
3. Write to disk incrementally
4. Use generators instead of lists

## Future Enhancements

### Planned Features

1. **Parallel Generation**: Multi-core generation for speed
2. **Active Learning**: Prioritize parameter regions that need more data
3. **Quality-Based Sampling**: Generate more examples in low-quality regions
4. **Cross-Validation Splits**: Automatic train/val/test split
5. **Data Augmentation**: Transpose, time-stretch generated MIDIs
6. **Advanced Validation**: Style consistency, theory compliance

### Extensibility

To add custom validation:

```python
class CustomValidator(MusicalCoherenceValidator):
    def validate_coherence(self, midi: mido.MidiFile) -> float:
        score = super().validate_coherence(midi)

        # Add custom checks
        custom_score = self._check_custom_metric(midi)

        # Combine scores
        return (score + custom_score) / 2

    def _check_custom_metric(self, midi: mido.MidiFile) -> float:
        # Your custom validation logic
        return 1.0
```

## Contributing

See main project README for contribution guidelines.

## License

MIT License - See main project LICENSE file.

## Author

**Agent 14** - Synthetic Training Data Generator
Part of the 35-agent Musical Program Synthesis system

## References

- Latin Hypercube Sampling: McKay et al., 1979
- XGBoost: Chen & Guestrin, 2016
- Musical Feature Extraction: Deep Feature Extractor (Agent 2)
- Parameter Registry: Universal Registry (Agent 3)

---

**Version**: 1.0.0
**Last Updated**: 2025-11-20
**Status**: Production Ready
