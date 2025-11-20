#!/usr/bin/env python3
"""
Example: Train a Single Parameter Model

This script demonstrates how to train a model for a single parameter
using the Model Training Specialist.

Usage:
    python train_single.py
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training import ModelTrainingSpecialist, TrainingConfig
from parameters.universal_registry import REGISTRY


def generate_synthetic_data(n_samples: int = 1000):
    """
    Generate synthetic training data for demonstration.

    In production, this would come from:
    - Agent 14 (Synthetic Training Data Generator)
    - Real MIDI analysis
    """
    print(f"Generating {n_samples} synthetic training examples...")

    training_data = []

    for i in range(n_samples):
        # Generate random features (in reality, these come from DeepFeatureExtractor)
        features = np.random.rand(135)  # 135 features for demo

        # Generate parameter value with some relationship to features
        # For harmony.voicing.spread (range 0.0 - 1.0)
        parameter_value = np.clip(
            0.3 + 0.4 * features[0] + 0.3 * features[1] + np.random.normal(0, 0.1),
            0.0, 1.0
        )

        training_data.append({
            'features': features,
            'parameter_value': parameter_value,
            'coherence_score': np.random.uniform(0.7, 0.95)
        })

    return training_data


def main():
    print("=" * 80)
    print("EXAMPLE: Train Single Parameter Model")
    print("=" * 80)

    # 1. Get parameter definition from registry
    param_name = 'harmony.voicing.spread'
    param_def = REGISTRY.get(param_name)

    if param_def is None:
        print(f"❌ Parameter not found: {param_name}")
        return

    print(f"\n📋 Parameter: {param_name}")
    print(f"   Type: {param_def.param_type.value}")
    print(f"   Description: {param_def.description}")
    print(f"   Range: [{param_def.min_value}, {param_def.max_value}]")

    # 2. Generate or load training data
    training_data = generate_synthetic_data(n_samples=1000)

    print(f"\n📊 Training Data:")
    print(f"   Examples: {len(training_data)}")
    print(f"   Features per example: {len(training_data[0]['features'])}")

    # 3. Configure training
    config = TrainingConfig(
        # Model parameters
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,

        # Training options
        enable_tuning=False,  # Set True for hyperparameter tuning
        early_stopping_rounds=10,

        # Quality thresholds
        min_r2=0.5,

        # Output
        save_plots=True,
        save_metrics=True,
        verbose=True
    )

    print(f"\n⚙️ Training Configuration:")
    print(f"   Estimators: {config.n_estimators}")
    print(f"   Max Depth: {config.max_depth}")
    print(f"   Learning Rate: {config.learning_rate}")
    print(f"   Hyperparameter Tuning: {config.enable_tuning}")

    # 4. Create trainer
    trainer = ModelTrainingSpecialist(config)

    # 5. Train model
    print(f"\n🚀 Starting training...")

    try:
        model, metrics = trainer.train_parameter_model(
            param_name=param_name,
            param_def=param_def,
            training_data=training_data,
            models_dir=Path('models/pretrained'),
            output_dir=Path('training_output')
        )

        # 6. Display results
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")

        print(f"\n📈 Performance Metrics:")
        print(f"   R² Score (test): {metrics.test_r2:.4f}")
        print(f"   MAE (test): {metrics.test_mae:.4f}")
        print(f"   RMSE (test): {metrics.test_rmse:.4f}")

        print(f"\n⏱️ Training Time: {metrics.training_time:.2f}s")

        print(f"\n✅ Quality Check: {'PASSED' if metrics.passed_quality_check else 'FAILED'}")
        print(f"   {metrics.quality_message}")

        print(f"\n🔍 Top 5 Most Important Features:")
        for feature, importance in metrics.top_features[:5]:
            print(f"   {feature}: {importance:.4f}")

        print(f"\n💾 Model saved to: {metrics.model_path}")

        # 7. Test prediction
        print(f"\n🧪 Testing predictions on 5 random examples:")
        for i in range(5):
            test_example = training_data[i]
            features = np.array([test_example['features']])
            prediction = model.predict(features)[0]
            actual = test_example['parameter_value']
            error = abs(prediction - actual)

            print(f"   Example {i+1}: predicted={prediction:.3f}, actual={actual:.3f}, error={error:.3f}")

        print(f"\n{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
