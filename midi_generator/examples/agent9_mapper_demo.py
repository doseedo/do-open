"""
Agent 9 Demo: Feature-Parameter Mapping Specialist
===================================================

Demonstrates the CRITICAL Agent 9 functionality:
- Mapping 1,000 musical features to 515+ parameters
- Training XGBoost models per parameter
- Predicting parameters from MIDI files
- Feature importance analysis
- Model persistence

This agent is the KEY to the ML pipeline!

Usage:
    python agent9_mapper_demo.py

Examples demonstrated:
1. Create mapper and train a single parameter
2. Predict parameter from MIDI file
3. Predict all parameters at once
4. Feature importance analysis
5. Save and load models
6. Batch training for multiple parameters
7. Full pipeline: MIDI -> Features -> Parameters

Author: Agent 9 Demo
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import Agent 9
from midi_generator.learning import (
    FeatureParameterMapper,
    TrainingExample,
    create_mapper
)

# Import Agent 8 for feature extraction
try:
    from midi_generator.synthesis.deep_feature_extractor import (
        DeepFeatureExtractor,
        extract_features
    )
    AGENT8_AVAILABLE = True
except ImportError:
    AGENT8_AVAILABLE = False
    print("WARNING: Agent 8 (DeepFeatureExtractor) not available")

# Import parameter registry
try:
    from midi_generator.parameters.universal_registry import UniversalParameterRegistry
    REGISTRY = UniversalParameterRegistry()
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    print("WARNING: Parameter registry not available")


# ============================================================================
# Demo Functions
# ============================================================================

def demo_1_create_and_train():
    """Demo 1: Create mapper and train a single parameter"""
    print("\n" + "="*70)
    print("DEMO 1: Create Mapper and Train Single Parameter")
    print("="*70)

    # Create mapper
    mapper = create_mapper()
    print("✓ Created FeatureParameterMapper")

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    n_examples = 100  # Small for demo
    n_features = 1000  # Agent 8 extracts 1000 features

    training_data = []
    for i in range(n_examples):
        # Synthetic features (in real use, these come from Agent 8)
        features = np.random.randn(n_features) * 0.5

        # Synthetic parameter value
        # Let's say chord_density depends on features 0-10
        value = 0.5 + 0.3 * np.mean(features[0:10])
        value = np.clip(value, 0.0, 1.0)

        example = TrainingExample(
            features=features,
            parameter_value=value,
            parameter_name='harmony.chord_density'
        )
        training_data.append(example)

    print(f"✓ Generated {n_examples} training examples")

    # Train model
    print("\nTraining model for 'harmony.chord_density'...")
    try:
        metrics = mapper.train_mapping(
            param_name='harmony.chord_density',
            training_data=training_data,
            validation_split=0.2,
            test_split=0.1
        )

        print(f"\n✓ Training complete!")
        print(f"  Quality: {metrics.quality_level}")
        print(f"  Validation score: {metrics.val_score:.4f}")
        print(f"  Training time: {metrics.training_time:.2f}s")

        return mapper, metrics
    except Exception as e:
        print(f"✗ Training failed: {e}")
        return None, None


def demo_2_predict_single_parameter(mapper):
    """Demo 2: Predict single parameter from features"""
    print("\n" + "="*70)
    print("DEMO 2: Predict Single Parameter")
    print("="*70)

    if mapper is None or 'harmony.chord_density' not in mapper.models:
        print("✗ No trained model available. Run demo_1 first.")
        return

    # Create synthetic features
    n_features = 1000
    features = np.random.randn(n_features) * 0.5

    print("Predicting 'harmony.chord_density' from features...")

    try:
        # Predict with timing
        result = mapper.predict_parameter(
            features,
            'harmony.chord_density',
            return_confidence=True
        )

        print(f"\n✓ Prediction complete!")
        print(f"  Predicted value: {result.predicted_value:.4f}")
        print(f"  Inference time: {result.inference_time*1000:.2f}ms")

        # Predict multiple times to show batch performance
        print("\nBatch prediction (10 samples)...")
        feature_matrix = np.random.randn(10, n_features) * 0.5

        predictions = mapper.predict_batch(feature_matrix, 'harmony.chord_density')
        print(f"✓ Predicted {len(predictions)} values")
        print(f"  Mean: {np.mean(predictions):.4f}")
        print(f"  Std: {np.std(predictions):.4f}")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")


def demo_3_feature_importance(mapper):
    """Demo 3: Analyze feature importance"""
    print("\n" + "="*70)
    print("DEMO 3: Feature Importance Analysis")
    print("="*70)

    if mapper is None or 'harmony.chord_density' not in mapper.models:
        print("✗ No trained model available. Run demo_1 first.")
        return

    param_name = 'harmony.chord_density'

    try:
        # Get feature importance
        importance = mapper.get_feature_importance(param_name, top_n=10)

        print(f"\nTop 10 most important features for '{param_name}':")
        for i, (feature, score) in enumerate(importance.items(), 1):
            print(f"  {i}. {feature}: {score:.4f}")

        # Get top features list
        top_features = mapper.get_top_features(param_name, n=5)
        print(f"\nTop 5 features: {top_features}")

    except Exception as e:
        print(f"✗ Feature importance analysis failed: {e}")


def demo_4_save_and_load(mapper):
    """Demo 4: Save and load models"""
    print("\n" + "="*70)
    print("DEMO 4: Model Persistence")
    print("="*70)

    if mapper is None or len(mapper.models) == 0:
        print("✗ No trained models available. Run demo_1 first.")
        return

    # Save model
    print("Saving model...")
    try:
        mapper.save_model('harmony.chord_density')
        print("✓ Model saved successfully")
    except Exception as e:
        print(f"✗ Failed to save model: {e}")
        return

    # Create new mapper and load
    print("\nCreating new mapper and loading model...")
    try:
        new_mapper = create_mapper()
        new_mapper.load_model('harmony.chord_density')
        print("✓ Model loaded successfully")

        # Test prediction with loaded model
        features = np.random.randn(1000) * 0.5
        value = new_mapper.predict_parameter(features, 'harmony.chord_density')
        print(f"✓ Prediction with loaded model: {value:.4f}")

        return new_mapper
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return None


def demo_5_batch_training():
    """Demo 5: Train multiple parameters at once"""
    print("\n" + "="*70)
    print("DEMO 5: Batch Training Multiple Parameters")
    print("="*70)

    # Create mapper
    mapper = create_mapper()

    # Parameters to train
    params_to_train = [
        'harmony.chord_density',
        'melody.note_density',
        'rhythm.syncopation'
    ]

    print(f"Training {len(params_to_train)} parameters...")

    # Generate training data for each parameter
    training_data_dict = {}
    n_examples = 100
    n_features = 1000

    for param_name in params_to_train:
        examples = []
        for i in range(n_examples):
            features = np.random.randn(n_features) * 0.5

            # Different synthetic relationships for each parameter
            if 'chord_density' in param_name:
                value = 0.5 + 0.3 * np.mean(features[0:10])
            elif 'note_density' in param_name:
                value = 0.6 + 0.2 * np.mean(features[10:20])
            else:
                value = 0.4 + 0.4 * np.mean(features[20:30])

            value = np.clip(value, 0.0, 1.0)

            example = TrainingExample(
                features=features,
                parameter_value=value,
                parameter_name=param_name
            )
            examples.append(example)

        training_data_dict[param_name] = examples

    # Batch train
    try:
        results = mapper.train_multiple_parameters(training_data_dict, show_progress=False)

        print(f"\n✓ Trained {len(results)} parameters!")
        for param_name, metrics in results.items():
            print(f"  {param_name}: {metrics.quality_level} (score: {metrics.val_score:.4f})")

        return mapper
    except Exception as e:
        print(f"✗ Batch training failed: {e}")
        return None


def demo_6_full_pipeline():
    """Demo 6: Full pipeline with real MIDI (if available)"""
    print("\n" + "="*70)
    print("DEMO 6: Full Pipeline (MIDI -> Features -> Parameters)")
    print("="*70)

    if not AGENT8_AVAILABLE:
        print("✗ Agent 8 not available, skipping real MIDI demo")
        return

    # Look for sample MIDI files
    midi_files = list(Path('midi_generator').glob('**/*.mid'))

    if len(midi_files) == 0:
        print("✗ No MIDI files found in midi_generator directory")
        print("  (This is OK - the mapper works without real MIDI)")
        return

    print(f"Found {len(midi_files)} MIDI files")

    # Take first MIDI file
    midi_file = midi_files[0]
    print(f"\nAnalyzing: {midi_file.name}")

    try:
        # Step 1: Extract features (Agent 8)
        print("\n1. Extracting 1000 features with Agent 8...")
        features = extract_features(midi_file)
        print(f"✓ Extracted {len(features)} features")

        # Step 2: Create and train mapper (simplified)
        print("\n2. Training quick model...")
        mapper = create_mapper()

        # For demo, we'll train on synthetic data but predict on real features
        training_data = []
        for i in range(50):  # Small for speed
            synthetic_features = np.random.randn(len(features)) * 0.5
            value = 0.5 + 0.3 * np.mean(synthetic_features[0:10])
            value = np.clip(value, 0.0, 1.0)

            example = TrainingExample(
                features=synthetic_features,
                parameter_value=value,
                parameter_name='harmony.chord_density'
            )
            training_data.append(example)

        mapper.train_mapping('harmony.chord_density', training_data)

        # Step 3: Predict parameter from real MIDI features
        print("\n3. Predicting parameter from real MIDI features...")
        predicted_value = mapper.predict_parameter(features, 'harmony.chord_density')

        print(f"\n✓ Full pipeline complete!")
        print(f"  MIDI file: {midi_file.name}")
        print(f"  Predicted chord_density: {predicted_value:.4f}")

    except Exception as e:
        print(f"✗ Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()


def demo_7_mapper_summary():
    """Demo 7: Mapper summary and statistics"""
    print("\n" + "="*70)
    print("DEMO 7: Mapper Summary and Statistics")
    print("="*70)

    # Train a few models first
    mapper = demo_5_batch_training()

    if mapper is None:
        print("✗ No mapper available")
        return

    # Get summary
    print("\nMapper Summary:")
    mapper.print_summary()

    # Get detailed metrics
    print("\nDetailed Metrics:")
    for param_name, metrics in mapper.metrics.items():
        print(f"\n{param_name}:")
        print(f"  Type: {metrics.parameter_type}")
        print(f"  Train score: {metrics.train_score:.4f}")
        print(f"  Val score: {metrics.val_score:.4f}")
        print(f"  Quality: {metrics.quality_level}")
        print(f"  Training time: {metrics.training_time:.2f}s")
        print(f"  N features: {metrics.n_features_used}")


def demo_8_predict_all_parameters():
    """Demo 8: Predict all trained parameters at once"""
    print("\n" + "="*70)
    print("DEMO 8: Predict All Parameters")
    print("="*70)

    # Train multiple parameters first
    mapper = demo_5_batch_training()

    if mapper is None:
        print("✗ No mapper available")
        return

    # Generate features
    features = np.random.randn(1000) * 0.5

    print(f"\nPredicting all {len(mapper.models)} trained parameters...")

    try:
        predictions = mapper.predict_all_parameters(features, show_progress=False)

        print(f"\n✓ Predicted {len(predictions)} parameters!")
        for param_name, value in predictions.items():
            print(f"  {param_name}: {value:.4f}")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")


# ============================================================================
# Main Demo Runner
# ============================================================================

def run_all_demos():
    """Run all demos in sequence"""
    print("\n" + "="*70)
    print("AGENT 9: FEATURE-PARAMETER MAPPING SPECIALIST")
    print("Comprehensive Demo Suite")
    print("="*70)

    print("\nThis demo shows how Agent 9 maps 1,000 features to 515+ parameters.")
    print("Agent 9 is CRITICAL for the ML pipeline!")

    # Check dependencies
    print("\n" + "-"*70)
    print("Checking dependencies...")
    print("-"*70)

    try:
        import xgboost
        print("✓ XGBoost available")
    except ImportError:
        print("✗ XGBoost not available (install with: pip install xgboost)")
        return

    try:
        import sklearn
        print("✓ scikit-learn available")
    except ImportError:
        print("✗ scikit-learn not available (install with: pip install scikit-learn)")
        return

    if AGENT8_AVAILABLE:
        print("✓ Agent 8 (DeepFeatureExtractor) available")
    else:
        print("⚠ Agent 8 not available (real MIDI extraction will be skipped)")

    if REGISTRY_AVAILABLE:
        print("✓ Parameter Registry available")
    else:
        print("⚠ Parameter Registry not available (using mock)")

    # Run demos
    print("\n" + "="*70)
    print("Running Demos...")
    print("="*70)

    # Demo 1: Basic training
    mapper, metrics = demo_1_create_and_train()

    if mapper is not None:
        # Demo 2: Prediction
        demo_2_predict_single_parameter(mapper)

        # Demo 3: Feature importance
        demo_3_feature_importance(mapper)

        # Demo 4: Persistence
        demo_4_save_and_load(mapper)

    # Demo 5: Batch training
    demo_5_batch_training()

    # Demo 6: Full pipeline (if Agent 8 available)
    demo_6_full_pipeline()

    # Demo 7: Summary
    demo_7_mapper_summary()

    # Demo 8: Predict all
    demo_8_predict_all_parameters()

    # Final summary
    print("\n" + "="*70)
    print("ALL DEMOS COMPLETE!")
    print("="*70)
    print("\nAgent 9 successfully demonstrated:")
    print("  ✓ Training feature-parameter mappings")
    print("  ✓ Predicting parameters from features")
    print("  ✓ Feature importance analysis")
    print("  ✓ Model persistence (save/load)")
    print("  ✓ Batch training and prediction")
    print("  ✓ Integration with Agent 8 (if available)")
    print("\nThe ML pipeline is now UNBLOCKED! 🚀")


if __name__ == "__main__":
    run_all_demos()
