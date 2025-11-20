<<<<<<< HEAD
# Synthetic Training Data Generator - Agent 14

## Overview

The Synthetic Training Data Generator is a comprehensive system for creating high-quality training datasets for new parameters in the Musical Program Synthesis system. It generates diverse, musically coherent MIDI examples with extracted features for training XGBoost models.
=======
# Model Training Specialist - Agent 15

Comprehensive XGBoost model training system for parameter prediction in the Musical Program Synthesis framework.

## Overview

The Model Training Specialist provides end-to-end training infrastructure for learning parameter values from musical features. Each parameter gets its own XGBoost model, trained to predict parameter values from 1000+ extracted features.

### Key Features

✅ **Automatic Pipeline**
- Automatic objective selection (regression/classification)
- Intelligent data splitting with stratification
- Early stopping for efficiency
- Comprehensive evaluation metrics

✅ **Optimization**
- Hyperparameter tuning (Grid Search / Random Search)
- Cross-validation support
- Quality threshold validation
- Learning curve analysis

✅ **Analysis**
- Feature importance ranking
- Top feature identification
- Model performance visualization
- Detailed training reports

✅ **Scalability**
- Batch training for multiple parameters
- Progress monitoring and logging
- Error handling and recovery
- Model versioning and metadata
>>>>>>> origin/claude/music-generation-agents-01Gi7dHdzZMrKvdMYFvonT1n

## Architecture

```
<<<<<<< HEAD
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
=======
training/
├── model_trainer.py          # Main training implementation
├── __init__.py              # Module interface
├── README.md                # This file
└── examples/                # Usage examples
    ├── train_single.py      # Train one parameter
    ├── train_batch.py       # Train multiple parameters
    └── evaluate_models.py   # Evaluate trained models
```

## Installation

### Requirements

```bash
pip install xgboost scikit-learn pandas numpy matplotlib joblib
```

### Optional Dependencies

```bash
pip install optuna  # For advanced hyperparameter tuning
```

## Usage

### 1. Train a Single Parameter

```python
from training import ModelTrainingSpecialist, TrainingConfig
from parameters.universal_registry import REGISTRY

# Get parameter definition
param_def = REGISTRY.get('harmony.voicing.spread')

# Configure training
config = TrainingConfig(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    enable_tuning=False
)

# Create trainer
trainer = ModelTrainingSpecialist(config)

# Load training data
training_data = [
    {
        'features': [0.1, 0.2, ...],  # 1000+ features
        'parameter_value': 0.5,        # Target value
        'coherence_score': 0.85        # Optional quality metric
    },
    # ... more examples
]

# Train model
model, metrics = trainer.train_parameter_model(
    param_name='harmony.voicing.spread',
    param_def=param_def,
    training_data=training_data,
    models_dir=Path('models/pretrained'),
    output_dir=Path('training_output')
)

# Print results
print(metrics.summary())
```

### 2. Batch Training

```python
from training import ModelTrainingSpecialist
from parameters.universal_registry import REGISTRY

# Get parameters to train
parameters = [
    ('harmony.voicing.spread', REGISTRY.get('harmony.voicing.spread')),
    ('melody.chromaticism.amount', REGISTRY.get('melody.chromaticism.amount')),
    ('rhythm.swing.amount', REGISTRY.get('rhythm.swing.amount')),
]

# Configure training
config = TrainingConfig(
    n_estimators=100,
    enable_tuning=True,
    tuning_method='grid'
)

trainer = ModelTrainingSpecialist(config)

# Batch train
results = trainer.train_batch(
    parameters=parameters,
    training_data_dir=Path('training_data'),
    models_dir=Path('models/pretrained'),
    output_dir=Path('training_output'),
    continue_on_error=True
)

# Print summary
print(results.summary())
```

### 3. Train All Parameters

```python
from training import train_all_parameters, TrainingConfig

config = TrainingConfig(
    n_estimators=150,
    max_depth=8,
    enable_tuning=True
)

results = train_all_parameters(
    training_data_dir=Path('training_data'),
    models_dir=Path('models/pretrained'),
    output_dir=Path('training_output'),
    config=config
)
```

### 4. Command-Line Interface

```bash
# Train single parameter
python -m training.model_trainer single \
    --param harmony.voicing.spread \
    --data-dir training_data \
    --models-dir models/pretrained \
    --output-dir output

# Train all parameters
python -m training.model_trainer all \
    --data-dir training_data \
    --models-dir models/pretrained \
    --output-dir output \
    --tune \
    --tuning-method random

# With custom hyperparameters
python -m training.model_trainer batch \
    --data-dir training_data \
    --n-estimators 200 \
    --max-depth 10 \
    --learning-rate 0.05 \
    --tune
```

## Training Data Format

Training data must be organized as follows:

```
training_data/
├── harmony_voicing_spread/
│   ├── metadata.json
│   └── data.pkl (or data.csv/data.json)
├── melody_chromaticism_amount/
│   ├── metadata.json
│   └── data.pkl
└── ...
```

### Data Format

Each training example must include:

```python
{
    'features': np.array([...]),      # 1000+ features from DeepFeatureExtractor
    'parameter_value': 0.5,           # Ground truth parameter value
    'coherence_score': 0.85,          # Optional: musical quality score
    'midi_file': 'path/to/file.mid'   # Optional: source MIDI file
}
>>>>>>> origin/claude/music-generation-agents-01Gi7dHdzZMrKvdMYFvonT1n
```

### Metadata Format

```json
{
<<<<<<< HEAD
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
=======
    "parameter_name": "harmony.voicing.spread",
    "parameter_type": "probability",
    "n_examples": 1000,
    "value_range": [0.0, 1.0],
    "feature_extractor_version": "1.0",
    "generation_date": "2025-01-20"
}
```

## Training Configuration

### TrainingConfig Parameters

```python
config = TrainingConfig(
    # Data splitting
    test_size=0.15,              # Test set size (15%)
    val_size=0.15,               # Validation set size (15%)
    random_state=42,             # Random seed
    stratify=True,               # Stratified splitting

    # XGBoost parameters
    n_estimators=100,            # Number of trees
    max_depth=6,                 # Maximum tree depth
    learning_rate=0.1,           # Learning rate
    subsample=0.8,               # Row subsampling
    colsample_bytree=0.8,        # Column subsampling
    min_child_weight=1,          # Minimum child weight
    gamma=0.0,                   # Minimum split loss
    reg_alpha=0.0,               # L1 regularization
    reg_lambda=1.0,              # L2 regularization

    # Early stopping
    early_stopping_rounds=10,    # Stop if no improvement

    # Hyperparameter tuning
    enable_tuning=False,         # Enable tuning
    tuning_method='grid',        # 'grid' or 'random'
    n_iter=50,                   # Iterations for random search
    cv_folds=3,                  # Cross-validation folds

    # Quality thresholds
    min_r2=0.5,                  # Minimum R² for regression
    min_accuracy=0.5,            # Minimum accuracy for classification
    min_f1=0.4,                  # Minimum F1 score

    # Output
    save_plots=True,             # Save visualization plots
    save_metrics=True,           # Save metrics to JSON
    verbose=True,                # Verbose output
    n_jobs=-1,                   # Parallel jobs (-1 = all cores)
    use_gpu=False                # Use GPU acceleration
)
```

## Output Files

### Per-Model Outputs

For each trained parameter:

```
models/pretrained/
├── harmony_voicing_spread.pkl              # Trained model
├── harmony_voicing_spread_metadata.json    # Model metadata
└── harmony_voicing_spread_encoder.pkl      # Label encoder (if categorical)

training_output/
├── harmony_voicing_spread_metrics.json     # Training metrics
├── harmony_voicing_spread_feature_importance.png
└── harmony_voicing_spread_predictions.png
```

### Batch Training Outputs

```
training_output/
├── batch_training_results.json    # Complete results
└── batch_training_summary.txt     # Human-readable summary
```

## Evaluation Metrics

### Regression Metrics

- **R² Score**: Coefficient of determination (target: > 0.5, ideal: > 0.7)
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **MAPE**: Mean Absolute Percentage Error

### Classification Metrics

- **Accuracy**: Overall accuracy (target: > 0.5, ideal: > 0.7)
- **F1 Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **AUC-ROC**: Area Under ROC Curve (binary classification)

## Quality Checks

Models must pass quality checks:

1. **Regression**: R² ≥ 0.5
2. **Classification**: Accuracy ≥ 0.5 AND F1 ≥ 0.4

If quality checks fail:
- Hyperparameter tuning is triggered (if enabled)
- Warning is issued in training report
- Model is still saved but flagged

## Feature Importance

Feature importance analysis identifies which extracted features are most predictive:

```python
# Top 10 features for harmony.voicing.spread
{
    'chroma_mean_0': 0.089,
    'spectral_centroid_mean': 0.067,
    'harmonic_complexity': 0.054,
    'interval_entropy': 0.048,
    ...
}
```

This helps:
- Understand what drives parameter values
- Identify redundant features
- Guide feature engineering
- Debug model behavior

## Advanced Usage

### Custom Feature Extractors

```python
from training import ModelTrainingSpecialist

class CustomTrainer(ModelTrainingSpecialist):
    def _get_feature_names(self):
        # Use your custom feature names
        return ['custom_feature_1', 'custom_feature_2', ...]

trainer = CustomTrainer(config)
```

### Custom Quality Checks

```python
def custom_quality_check(metrics, param_def):
    if metrics.test_r2 is not None:
        # Custom regression threshold
        return metrics.test_r2 >= 0.6, "Custom check"
    else:
        # Custom classification threshold
        return metrics.test_accuracy >= 0.7, "Custom check"

# Override in trainer
trainer._check_quality = custom_quality_check
```

### GPU Acceleration

```python
config = TrainingConfig(
    use_gpu=True,
    n_estimators=500,  # Can train more trees with GPU
    max_depth=10
)
```

Requires: `pip install xgboost[gpu]`

## Performance Optimization

### Tips for Faster Training

1. **Reduce Features**: Use feature selection to identify most important features
2. **Subsample Data**: For initial experiments, use subset of training data
3. **Parallel Training**: Train multiple parameters in parallel
4. **Use GPU**: For large datasets and deep trees
5. **Adjust Early Stopping**: Lower `early_stopping_rounds` for faster convergence

### Memory Optimization

For large datasets:

```python
config = TrainingConfig(
    subsample=0.5,           # Use 50% of data per tree
    colsample_bytree=0.5,    # Use 50% of features per tree
    max_depth=6,             # Limit tree depth
)
>>>>>>> origin/claude/music-generation-agents-01Gi7dHdzZMrKvdMYFvonT1n
```

## Troubleshooting

<<<<<<< HEAD
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
=======
### Issue: Low R² / Accuracy

**Solutions:**
- Enable hyperparameter tuning
- Collect more training data
- Check for data quality issues
- Verify feature extraction is correct
- Consider ensemble models

### Issue: Overfitting (train >> test)

**Solutions:**
- Increase regularization (`reg_alpha`, `reg_lambda`)
- Reduce `max_depth`
- Increase `min_child_weight`
- Add more training data
- Enable early stopping

### Issue: Training Too Slow

**Solutions:**
- Reduce `n_estimators`
- Reduce dataset size
- Use fewer features
- Enable GPU acceleration
- Disable hyperparameter tuning

### Issue: Out of Memory

**Solutions:**
- Reduce `subsample` and `colsample_bytree`
- Train parameters individually instead of batch
- Use smaller `max_depth`
- Close other applications

## Integration with System

### 1. Generate Training Data

```python
# Use Agent 14 (Synthetic Training Generator)
from synthesis.training_data_generator import generate_training_data

generate_training_data(
    param_name='harmony.voicing.spread',
    n_samples=1000,
    output_dir=Path('training_data')
)
```

### 2. Train Models

```python
# Use Agent 15 (Model Training Specialist)
from training import train_all_parameters

results = train_all_parameters(
    training_data_dir=Path('training_data'),
    models_dir=Path('models/pretrained')
)
```

### 3. Use Models for Prediction

```python
# Use Agent 16 (Parameter Synthesizer)
from synthesis.parameter_synthesizer import ParameterSynthesizer

synthesizer = ParameterSynthesizer()
predicted_params = synthesizer.predict_from_midi('input.mid')
```

## Performance Targets

### Minimum Acceptable
- **Regression**: R² > 0.5
- **Classification**: Accuracy > 0.5, F1 > 0.4
- **Training Time**: < 5 minutes per parameter

### Target Performance
- **Regression**: R² > 0.7
- **Classification**: Accuracy > 0.7, F1 > 0.6
- **Training Time**: < 2 minutes per parameter

### Excellent Performance
- **Regression**: R² > 0.85
- **Classification**: Accuracy > 0.85, F1 > 0.8
- **Training Time**: < 1 minute per parameter

## Roadmap

### Phase 1 (Current)
✅ Basic training pipeline
✅ Hyperparameter tuning
✅ Feature importance analysis
✅ Batch training support

### Phase 2 (Next)
- [ ] Advanced feature selection
- [ ] Ensemble model support
- [ ] Active learning integration
- [ ] Online/incremental learning

### Phase 3 (Future)
- [ ] Neural network models
- [ ] Multi-task learning
- [ ] Transfer learning
- [ ] AutoML integration

## References

- XGBoost Documentation: https://xgboost.readthedocs.io/
- Scikit-learn Model Selection: https://scikit-learn.org/stable/model_selection.html
- Feature Engineering Guide: https://www.kaggle.com/learn/feature-engineering

## Authors

**Agent 15 - Model Training Specialist**

Part of the 35-Agent Musical Program Synthesis System

## License

MIT License

---

For questions or issues, see the main project documentation.
>>>>>>> origin/claude/music-generation-agents-01Gi7dHdzZMrKvdMYFvonT1n
