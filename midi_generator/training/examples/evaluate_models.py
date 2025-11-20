#!/usr/bin/env python3
"""
Example: Evaluate Trained Models

This script demonstrates how to load and evaluate trained models.

Usage:
    python evaluate_models.py
"""

import sys
from pathlib import Path
import json
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import joblib
except ImportError:
    import pickle as joblib


def load_model(model_path: Path):
    """Load a trained model"""
    if model_path.exists():
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")


def load_metadata(metadata_path: Path):
    """Load model metadata"""
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        return {}


def evaluate_model(model, test_data: list):
    """Evaluate model on test data"""
    features = np.array([ex['features'] for ex in test_data])
    actuals = np.array([ex['parameter_value'] for ex in test_data])

    predictions = model.predict(features)

    # Calculate metrics
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    r2 = r2_score(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))

    return {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'predictions': predictions,
        'actuals': actuals
    }


def main():
    print("=" * 80)
    print("EXAMPLE: Evaluate Trained Models")
    print("=" * 80)

    models_dir = Path('models/pretrained')

    # Find all trained models
    model_files = list(models_dir.glob('*.pkl'))
    model_files = [f for f in model_files if not f.name.endswith('_encoder.pkl')]

    if not model_files:
        print(f"\n❌ No trained models found in: {models_dir}")
        print(f"   Run train_single.py or train_batch.py first!")
        return

    print(f"\n📊 Found {len(model_files)} trained models")

    # Evaluate each model
    results = []

    for model_path in model_files:
        param_name = model_path.stem.replace('_', '.')
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

        print(f"\n{'='*80}")
        print(f"Model: {param_name}")
        print(f"{'='*80}")

        try:
            # Load model
            model = load_model(model_path)
            print(f"✅ Model loaded: {model_path.name}")

            # Load metadata
            metadata = load_metadata(metadata_path)

            if metadata:
                print(f"\n📋 Model Information:")
                print(f"   Parameter: {metadata.get('param_name', 'N/A')}")
                print(f"   Type: {metadata.get('param_type', 'N/A')}")
                print(f"   Description: {metadata.get('param_description', 'N/A')}")
                print(f"   Timestamp: {metadata.get('timestamp', 'N/A')}")

                if 'metrics' in metadata:
                    print(f"\n📈 Training Metrics:")
                    metrics = metadata['metrics']
                    if metrics.get('test_r2') is not None:
                        print(f"   R² Score: {metrics['test_r2']:.4f}")
                        print(f"   MAE: {metrics['test_mae']:.4f}")
                    if metrics.get('test_accuracy') is not None:
                        print(f"   Accuracy: {metrics['test_accuracy']:.4f}")
                        print(f"   F1 Score: {metrics['test_f1']:.4f}")

                    quality = "✅ PASSED" if metrics.get('passed_quality') else "⚠️ NEEDS IMPROVEMENT"
                    print(f"   Quality Check: {quality}")

            # Generate test data
            print(f"\n🧪 Generating test data for evaluation...")
            test_data = []
            for i in range(100):
                features = np.random.rand(135)
                # Use a simple relationship for synthetic data
                parameter_value = np.clip(0.3 + 0.4 * features[0] + 0.3 * features[1], 0.0, 1.0)
                test_data.append({
                    'features': features,
                    'parameter_value': parameter_value
                })

            # Evaluate
            eval_results = evaluate_model(model, test_data)

            print(f"\n📊 Evaluation Results (100 test samples):")
            print(f"   R² Score: {eval_results['r2']:.4f}")
            print(f"   MAE: {eval_results['mae']:.4f}")
            print(f"   RMSE: {eval_results['rmse']:.4f}")

            # Show some predictions
            print(f"\n🔍 Sample Predictions:")
            for i in range(5):
                pred = eval_results['predictions'][i]
                actual = eval_results['actuals'][i]
                error = abs(pred - actual)
                print(f"   Sample {i+1}: pred={pred:.3f}, actual={actual:.3f}, error={error:.3f}")

            results.append({
                'param_name': param_name,
                'r2': eval_results['r2'],
                'mae': eval_results['mae'],
                'rmse': eval_results['rmse']
            })

        except Exception as e:
            print(f"❌ Error evaluating model: {e}")

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")

        print(f"\n📊 All Models:")
        for result in sorted(results, key=lambda x: x['r2'], reverse=True):
            print(f"   {result['param_name']:40s}  R²={result['r2']:.4f}  MAE={result['mae']:.4f}")

        avg_r2 = np.mean([r['r2'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])

        print(f"\n📈 Average Performance:")
        print(f"   Average R²: {avg_r2:.4f}")
        print(f"   Average MAE: {avg_mae:.4f}")

        # Best and worst
        best = max(results, key=lambda x: x['r2'])
        worst = min(results, key=lambda x: x['r2'])

        print(f"\n🏆 Best Model:")
        print(f"   {best['param_name']} (R² = {best['r2']:.4f})")

        print(f"\n⚠️ Needs Improvement:")
        print(f"   {worst['param_name']} (R² = {worst['r2']:.4f})")

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}")


if __name__ == '__main__':
    main()
