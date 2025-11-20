# Quality Metrics Dashboard - Agent 31

**Real-time quality monitoring for the Musical Program Synthesis system**

## Overview

The Quality Metrics Dashboard provides comprehensive monitoring of the self-expanding music generation system. It tracks performance of all 165+ XGBoost models (one per parameter) and monitors reconstruction quality as the system grows toward 515+ and eventually 800+ parameters.

## Features

### 1. Per-Parameter R² Tracking
- Monitors R² scores for each XGBoost model
- Tracks MAE, RMSE, and correlation metrics
- Maintains historical trends (up to 1000 data points per parameter)
- Quality classification: Excellent (>0.90), Good (>0.75), Acceptable (>0.60), Poor (<0.60)

### 2. Reconstruction Quality Metrics
- MIDI → features → parameters → MIDI pipeline monitoring
- Note accuracy, rhythm accuracy, harmony accuracy
- Error tracking (missing notes, extra notes, pitch errors, timing errors)
- Parameter recovery rate

### 3. Real-Time Monitoring
- Live dashboard updates
- Automatic anomaly detection (statistical outliers)
- Trend analysis (improving, stable, declining, volatile)
- System health score (0-100)

### 4. Alerting System
- Severity levels: INFO, WARNING, ERROR, CRITICAL
- Automatic alerts for:
  - R² scores below threshold
  - Declining trends
  - Poor reconstruction quality
  - Detected anomalies

### 5. Visualization
- Parameter history plots (R², MAE over time)
- System health dashboard (4-panel overview)
- Category comparison (harmony, rhythm, melody, etc.)
- Trend lines and quality thresholds

### 6. Reporting
- Full system reports
- Per-parameter summaries
- Category-level analysis
- Improvement opportunity identification

## Quick Start

```python
from midi_generator.monitoring import QualityMetricsDashboard

# Create dashboard
dashboard = QualityMetricsDashboard(
    storage_dir="./monitoring_data",
    alert_threshold_r2=0.60,
    enable_visualization=True
)

# Register parameters from the universal registry
dashboard.register_parameter(
    parameter_name="voicing_type",
    parameter_path="harmony.voicing.type",
    category="harmony"
)

# Update metrics after training/validation
dashboard.update_parameter_metrics(
    parameter_path="harmony.voicing.type",
    r2_score=0.85,
    mae=0.12,
    rmse=0.15,
    training_samples=1000,
    training_time=45.2,
    inference_time=2.5
)

# Monitor reconstruction quality
dashboard.update_reconstruction_metrics(
    reconstruction_accuracy=0.87,
    parameter_recovery_rate=0.92,
    note_accuracy=0.95,
    rhythm_accuracy=0.88,
    harmony_accuracy=0.85
)

# Generate reports
health = dashboard.compute_system_health()
print(f"System Health: {health.health_score:.1f}/100")

report = dashboard.generate_report(report_type="full")
print(report)

# Create visualizations
dashboard.plot_system_health()
dashboard.plot_parameter_history("harmony.voicing.type")
dashboard.plot_category_comparison()

# Identify improvement opportunities
opportunities = dashboard.identify_improvement_opportunities(top_n=10)
for opp in opportunities:
    print(f"{opp['parameter']}: R²={opp['current_r2']:.3f}")
    for rec in opp['recommendations']:
        print(f"  → {rec}")

# Save state for persistence
dashboard.save_state()
```

## Integration with Existing System

### With Parameter Registry

```python
from midi_generator.parameters.universal_registry import UniversalParameterRegistry
from midi_generator.monitoring import QualityMetricsDashboard

# Load registry
registry = UniversalParameterRegistry()

# Create dashboard
dashboard = QualityMetricsDashboard()

# Auto-register all parameters
for param_path, param_def in registry.parameters.items():
    dashboard.register_parameter(
        parameter_name=param_def.name,
        parameter_path=param_path,
        category=param_def.category.value if param_def.category else "unknown"
    )
```

### With XGBoost Training Pipeline

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from midi_generator.monitoring import QualityMetricsDashboard

dashboard = QualityMetricsDashboard()

def train_parameter_model(parameter_path, X, y):
    """Train XGBoost model and update dashboard"""

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    # Train model
    import time
    start_time = time.time()

    model = xgb.XGBRegressor(n_estimators=100, max_depth=6)
    model.fit(X_train, y_train)

    training_time = time.time() - start_time

    # Validate
    y_pred = model.predict(X_val)

    r2 = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = mean_squared_error(y_val, y_pred, squared=False)

    # Get feature importance
    feature_importance = dict(zip(
        [f"feature_{i}" for i in range(X.shape[1])],
        model.feature_importances_
    ))

    # Update dashboard
    dashboard.update_parameter_metrics(
        parameter_path=parameter_path,
        r2_score=r2,
        mae=mae,
        rmse=rmse,
        training_samples=len(X_train),
        training_time=training_time,
        feature_importance=feature_importance
    )

    return model
```

### With MIDI Reconstruction Pipeline

```python
from midi_generator.monitoring import QualityMetricsDashboard

dashboard = QualityMetricsDashboard()

def evaluate_reconstruction(original_midi, reconstructed_midi):
    """Evaluate MIDI reconstruction quality"""

    # Your existing comparison logic here
    # ...

    # Update dashboard
    dashboard.update_reconstruction_metrics(
        reconstruction_accuracy=overall_accuracy,
        parameter_recovery_rate=param_recovery,
        note_accuracy=note_acc,
        rhythm_accuracy=rhythm_acc,
        harmony_accuracy=harmony_acc,
        errors={
            'missing_notes': missing_count,
            'extra_notes': extra_count,
            'pitch_errors': pitch_errors,
            'timing_errors': timing_errors
        }
    )
```

## Architecture

### Modular Design
- **One XGBoost model per parameter** - Each parameter has independent tracking
- **No retraining needed** - Adding new parameters doesn't affect existing ones
- **Scalable** - Designed to grow from 165 → 515 → 800+ parameters

### Data Structures

```python
# Per-parameter tracking
ParameterMetrics:
    - current_r2, mae, rmse
    - r2_history (deque, max 1000 points)
    - quality_level (excellent/good/acceptable/poor/critical)
    - trend_direction (improving/stable/declining/volatile)
    - feature_importance_scores
    - training statistics

# System-wide health
SystemHealthMetrics:
    - total_parameters, trained_parameters
    - model quality distribution
    - average/median/min/max R² scores
    - reconstruction accuracy
    - active alerts
    - health_score (0-100)

# Quality alerts
QualityAlert:
    - severity (info/warning/error/critical)
    - category, parameter_name
    - message, details
    - timestamp, resolution status
```

## Metrics Explained

### R² Score (Coefficient of Determination)
- **Range**: -∞ to 1.0 (typically 0.0 to 1.0)
- **Excellent**: > 0.90
- **Good**: > 0.75
- **Acceptable**: > 0.60
- **Poor**: < 0.60

### MAE (Mean Absolute Error)
- Average absolute difference between predicted and actual values
- Lower is better
- Same units as the parameter being predicted

### RMSE (Root Mean Square Error)
- Square root of average squared differences
- More sensitive to outliers than MAE
- Lower is better

### Reconstruction Accuracy
- **1.0**: Perfect reconstruction
- **> 0.90**: Excellent - minimal perceptual difference
- **> 0.75**: Good - minor differences
- **< 0.75**: Needs improvement

### Health Score (0-100)
Weighted combination:
- 40% average R² score
- 30% reconstruction accuracy
- 30% model coverage (trained/total)
- Penalties for alerts and poor models

## Alert Thresholds

Default thresholds (configurable):
- **R² below 0.60**: WARNING
- **R² below 0.40**: CRITICAL
- **Reconstruction < 0.70**: WARNING
- **Detected anomaly (>2.5σ)**: WARNING

## Anomaly Detection

Statistical outlier detection:
- Sliding window analysis (default: 50 measurements)
- Z-score threshold (default: 2.5 standard deviations)
- Focuses on sudden performance drops
- Marks parameters for investigation

## Trend Analysis

Trend classification:
- **Improving**: Positive slope, low variance
- **Stable**: Minimal slope, low variance
- **Declining**: Negative slope
- **Volatile**: High variance regardless of slope

## File Persistence

Dashboard state is automatically saved to JSON:
- `dashboard_state.json`: Full state snapshot
- Last 100 measurements per parameter
- System health history
- Active alerts

## Visualization Output

Generated plots (PNG format):
1. **System Health Dashboard** (2×2 grid):
   - Average R² over time
   - Model quality distribution
   - Reconstruction accuracy
   - Overall health score

2. **Parameter History**:
   - R² or MAE over time
   - Trend line
   - Quality thresholds

3. **Category Comparison**:
   - Box plots by category
   - Average R² bar chart

## Best Practices

1. **Regular Updates**: Update metrics after each training/validation cycle
2. **Save State**: Call `dashboard.save_state()` periodically
3. **Monitor Alerts**: Check `dashboard.get_active_alerts()` regularly
4. **Review Trends**: Look for declining or volatile parameters
5. **Act on Opportunities**: Use `identify_improvement_opportunities()` to prioritize work

## Performance Considerations

- **Memory**: Deque limits prevent unbounded growth (max 1000 points per metric)
- **Storage**: JSON state files typically < 10MB for 500 parameters
- **Visualization**: Disabled by default if matplotlib unavailable
- **Statistics**: Graceful fallback if numpy/scipy unavailable

## Dependencies

### Required
- Python 3.7+
- dataclasses (built-in Python 3.7+)

### Optional
- numpy: Statistical calculations (fallback available)
- scipy: Advanced statistics (fallback available)
- matplotlib: Visualization (can be disabled)

## Example Output

```
📊 Quality Metrics Dashboard initialized
   Storage: ./monitoring_data
   Visualization: enabled
   Tracking: 165 parameters

SYSTEM HEALTH
--------------------------------------------------------------------------------
Overall Health Score: 87.3/100
Total Parameters: 165
Trained Models: 142 (86.1%)

Model Quality Distribution:
  Excellent (R²>0.90):  45 (31.7%)
  Good (R²>0.75):       72 (50.7%)
  Poor (R²<0.60):       12 (8.5%)

R² Statistics:
  Average: 0.812
  Median:  0.825
  Range:   [0.423, 0.947]

Reconstruction: 87.4% accuracy (523 samples)
Alerts: 3 active (0 critical)
```

## Future Enhancements

Planned for future releases:
- [ ] Web-based dashboard (Flask/FastAPI)
- [ ] Real-time streaming updates
- [ ] Comparative analysis (before/after changes)
- [ ] Export to Prometheus/Grafana
- [ ] Email/Slack alert integration
- [ ] A/B testing framework
- [ ] Parameter correlation analysis
- [ ] Automated retraining triggers

## Contributing

This dashboard is part of the 35-agent Musical Program Synthesis system. See the main project README for contribution guidelines.

## License

MIT License - See project root for details

## Author

Agent 31 - Quality Metrics Dashboard
Part of the Self-Expanding Inverse Music Generation System

## Related Documentation

- Agent 1: Parameter Auditor (`midi_generator/audit/parameter_auditor.py`)
- Agent 3: Universal Parameter Registry (`midi_generator/parameters/universal_registry.py`)
- Agent 16: Dataset Analysis Engine (`midi_generator/analysis/dataset_analyzer.py`)
