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

## Architecture

```
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
```

### Metadata Format

```json
{
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
```

## Troubleshooting

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
