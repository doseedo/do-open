#!/usr/bin/env python3
"""
Example: Batch Train Multiple Parameters

This script demonstrates how to train models for multiple parameters
simultaneously using batch training.

Usage:
    python train_batch.py
"""

import sys
from pathlib import Path
import numpy as np
import json

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from training import ModelTrainingSpecialist, TrainingConfig
from parameters.universal_registry import REGISTRY


def generate_training_data_for_parameter(param_name: str, param_def, n_samples: int = 500):
    """Generate synthetic training data for a specific parameter"""
    training_data = []

    for i in range(n_samples):
        features = np.random.rand(135)

        # Generate parameter value based on type
        if param_def.param_type.value in ['continuous', 'probability']:
            # Use features to generate correlated values
            parameter_value = np.clip(
                param_def.min_value +
                (param_def.max_value - param_def.min_value) * (
                    0.3 * features[0] +
                    0.3 * features[1] +
                    0.2 * features[2] +
                    0.2 * np.random.rand()
                ),
                param_def.min_value,
                param_def.max_value
            )

        elif param_def.param_type.value == 'boolean':
            # Generate boolean based on features
            parameter_value = bool(features[0] + features[1] > 1.0)

        elif param_def.param_type.value == 'categorical':
            # Random categorical choice
            parameter_value = np.random.choice(param_def.options)

        elif param_def.param_type.value == 'integer':
            # Generate integer
            parameter_value = int(np.clip(
                param_def.min_value +
                (param_def.max_value - param_def.min_value) * features[0],
                param_def.min_value,
                param_def.max_value
            ))

        else:
            parameter_value = param_def.default_value

        training_data.append({
            'features': features,
            'parameter_value': parameter_value,
            'coherence_score': np.random.uniform(0.7, 0.95)
        })

    return training_data


def save_training_data(param_name: str, training_data: list, output_dir: Path):
    """Save training data to disk"""
    param_dir = output_dir / param_name.replace('.', '_')
    param_dir.mkdir(parents=True, exist_ok=True)

    # Save data
    import pickle
    with open(param_dir / 'data.pkl', 'wb') as f:
        pickle.dump(training_data, f)

    # Save metadata
    metadata = {
        'parameter_name': param_name,
        'n_examples': len(training_data),
        'n_features': len(training_data[0]['features']),
        'value_range': [
            float(min(ex['parameter_value'] for ex in training_data)),
            float(max(ex['parameter_value'] for ex in training_data))
        ]
    }

    with open(param_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def main():
    print("=" * 80)
    print("EXAMPLE: Batch Train Multiple Parameters")
    print("=" * 80)

    # 1. Select parameters to train
    param_names = [
        'harmony.voicing.spread',
        'harmony.voicing.density',
        'melody.chromaticism.amount',
        'rhythm.swing.amount',
        'rhythm.syncopation.probability'
    ]

    print(f"\n📋 Parameters to train: {len(param_names)}")
    for name in param_names:
        print(f"   - {name}")

    # 2. Generate and save training data for each parameter
    training_data_dir = Path('training_data_batch_example')
    training_data_dir.mkdir(exist_ok=True)

    parameters = []
    for param_name in param_names:
        param_def = REGISTRY.get(param_name)
        if param_def is None:
            print(f"⚠️ Skipping unknown parameter: {param_name}")
            continue

        print(f"\n📊 Generating training data for: {param_name}")
        training_data = generate_training_data_for_parameter(
            param_name, param_def, n_samples=500
        )

        # Save to disk
        save_training_data(param_name, training_data, training_data_dir)

        parameters.append((param_name, param_def))

    # 3. Configure training
    config = TrainingConfig(
        # Model parameters
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,

        # Training options
        enable_tuning=False,  # Set True for hyperparameter tuning
        tuning_method='grid',
        early_stopping_rounds=10,

        # Quality thresholds
        min_r2=0.5,
        min_accuracy=0.5,

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

    # 5. Batch train
    print(f"\n🚀 Starting batch training...")

    try:
        results = trainer.train_batch(
            parameters=parameters,
            training_data_dir=training_data_dir,
            models_dir=Path('models/pretrained'),
            output_dir=Path('training_output_batch'),
            continue_on_error=True
        )

        # 6. Display results
        print(f"\n{'='*80}")
        print("BATCH TRAINING COMPLETE")
        print(f"{'='*80}")

        print(f"\n📊 Summary:")
        print(f"   Total Parameters: {results.total_parameters}")
        print(f"   Successful: {results.successful}")
        print(f"   Failed: {results.failed}")
        print(f"   Total Time: {results.total_time:.2f}s ({results.total_time/60:.2f}m)")

        if results.successful > 0:
            print(f"\n✅ Successfully Trained Models:")
            for param_name, metrics in results.results.items():
                quality = "✅" if metrics.passed_quality_check else "⚠️"
                if metrics.test_r2 is not None:
                    print(f"   {quality} {param_name}")
                    print(f"      R² = {metrics.test_r2:.4f}, MAE = {metrics.test_mae:.4f}")
                else:
                    print(f"   {quality} {param_name}")
                    print(f"      Accuracy = {metrics.test_accuracy:.4f}, F1 = {metrics.test_f1:.4f}")

        if results.failed > 0:
            print(f"\n❌ Failed Models:")
            for param_name, error in results.errors.items():
                print(f"   {param_name}: {error}")

        # 7. Average performance
        if results.successful > 0:
            avg_r2 = np.mean([
                m.test_r2 for m in results.results.values()
                if m.test_r2 is not None
            ])
            avg_accuracy = np.mean([
                m.test_accuracy for m in results.results.values()
                if m.test_accuracy is not None
            ])

            print(f"\n📈 Average Performance:")
            if not np.isnan(avg_r2):
                print(f"   Average R² (regression models): {avg_r2:.4f}")
            if not np.isnan(avg_accuracy):
                print(f"   Average Accuracy (classification models): {avg_accuracy:.4f}")

        # 8. Quality check summary
        passed = sum(1 for m in results.results.values() if m.passed_quality_check)
        print(f"\n✅ Quality Checks:")
        print(f"   Passed: {passed}/{results.successful} ({100*passed/results.successful:.1f}%)")

        print(f"\n{'='*80}")
        print("SUCCESS!")
        print(f"{'='*80}")

        print(f"\n💡 Next Steps:")
        print(f"   1. Review training metrics in: training_output_batch/")
        print(f"   2. Examine feature importance plots")
        print(f"   3. Use trained models with ParameterSynthesizer")
        print(f"   4. If quality is low, enable hyperparameter tuning")

    except Exception as e:
        print(f"\n❌ Batch training failed: {e}")
        raise


if __name__ == '__main__':
    main()
