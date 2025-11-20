"""
Example Usage Scripts for Synthetic Training Data Generator
============================================================

Demonstrates various use cases and best practices for generating
synthetic training data for music generation parameters.

Author: Agent 14
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
from midi_generator.training import (
    SyntheticTrainingDataGenerator,
    BatchTrainingDataGenerator,
    MusicalCoherenceValidator,
    ParameterSpaceSampler,
)
from midi_generator.parameters.universal_registry import (
    UniversalParameterRegistry,
    ParameterDefinition,
    ParameterType,
    ParameterCategory,
)


# ============================================================================
# EXAMPLE 1: Basic Usage
# ============================================================================

def example_basic_generation():
    """
    Basic example: Generate 100 training examples for a single parameter
    """
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Training Data Generation")
    print("="*60 + "\n")

    # Initialize components
    registry = UniversalParameterRegistry()
    generator = SyntheticTrainingDataGenerator(
        registry=registry,
        output_root=Path('example_training_data')
    )

    # Get a parameter (if available in registry)
    # For this example, we'll create a mock parameter
    param_name = "example.harmony.voicing_density"
    param_def = ParameterDefinition(
        name="voicing_density",
        full_path=param_name,
        description="Density of chord voicings (1.0 = dense, 0.0 = sparse)",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.HARMONY
    )

    # Generate training data
    print(f"Generating data for: {param_name}")
    print(f"Type: {param_def.param_type.value}")
    print(f"Range: [{param_def.min_value}, {param_def.max_value}]")
    print()

    try:
        training_data = generator.generate_training_data(
            param_name=param_name,
            param_def=param_def,
            n_examples=100,  # Small number for example
            min_coherence=0.5
        )

        print(f"\n✅ SUCCESS: Generated {len(training_data)} examples")

        # Analyze results
        coherence_scores = [ex.coherence_score for ex in training_data]
        print(f"\nCoherence Statistics:")
        print(f"  Mean: {np.mean(coherence_scores):.3f}")
        print(f"  Min: {np.min(coherence_scores):.3f}")
        print(f"  Max: {np.max(coherence_scores):.3f}")

        # Show parameter value distribution
        param_values = [ex.parameter_value for ex in training_data]
        print(f"\nParameter Value Distribution:")
        print(f"  Mean: {np.mean(param_values):.3f}")
        print(f"  Std: {np.std(param_values):.3f}")
        print(f"  Range: [{np.min(param_values):.3f}, {np.max(param_values):.3f}]")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXAMPLE 2: Genre-Balanced Generation
# ============================================================================

def example_genre_balanced_generation():
    """
    Generate genre-balanced training dataset
    """
    print("\n" + "="*60)
    print("EXAMPLE 2: Genre-Balanced Training Data Generation")
    print("="*60 + "\n")

    # Initialize
    registry = UniversalParameterRegistry()
    generator = SyntheticTrainingDataGenerator(
        registry=registry,
        output_root=Path('example_training_data_balanced')
    )

    # Define parameter
    param_name = "example.rhythm.swing_intensity"
    param_def = ParameterDefinition(
        name="swing_intensity",
        full_path=param_name,
        description="Intensity of swing feel (0.0 = straight, 1.0 = heavy swing)",
        param_type=ParameterType.PROBABILITY,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0,
        category=ParameterCategory.RHYTHM
    )

    print(f"Generating genre-balanced data for: {param_name}")
    print(f"Genres: swing, bebop, modal, bossa_nova")
    print()

    try:
        # Generate balanced dataset
        training_data = generator.generate_balanced_dataset(
            param_name=param_name,
            param_def=param_def,
            n_per_genre=25,  # 25 per genre = 200 total
            genres=['swing', 'bebop', 'modal', 'bossa_nova']
        )

        print(f"\n✅ SUCCESS: Generated {len(training_data)} examples")

        # Analyze genre distribution
        from collections import Counter
        genre_counts = Counter([ex.genre for ex in training_data if ex.genre])

        print(f"\nGenre Distribution:")
        for genre, count in genre_counts.items():
            print(f"  {genre}: {count} examples")

        # Analyze per-genre coherence
        print(f"\nPer-Genre Coherence:")
        for genre in genre_counts.keys():
            genre_examples = [ex for ex in training_data if ex.genre == genre]
            avg_coherence = np.mean([ex.coherence_score for ex in genre_examples])
            print(f"  {genre}: {avg_coherence:.3f}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXAMPLE 3: Batch Generation for Multiple Parameters
# ============================================================================

def example_batch_generation():
    """
    Generate training data for multiple parameters in batch
    """
    print("\n" + "="*60)
    print("EXAMPLE 3: Batch Generation for Multiple Parameters")
    print("="*60 + "\n")

    # Initialize batch generator
    batch_gen = BatchTrainingDataGenerator()

    # Define multiple parameters
    parameters = {
        "example.harmony.voicing_density": ParameterDefinition(
            name="voicing_density",
            full_path="example.harmony.voicing_density",
            description="Chord voicing density",
            param_type=ParameterType.CONTINUOUS,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            category=ParameterCategory.HARMONY
        ),
        "example.melody.chromaticism": ParameterDefinition(
            name="chromaticism",
            full_path="example.melody.chromaticism",
            description="Amount of chromatic notes in melody",
            param_type=ParameterType.PROBABILITY,
            default_value=0.3,
            min_value=0.0,
            max_value=1.0,
            category=ParameterCategory.MELODY
        ),
        "example.rhythm.syncopation": ParameterDefinition(
            name="syncopation",
            full_path="example.rhythm.syncopation",
            description="Degree of rhythmic syncopation",
            param_type=ParameterType.CONTINUOUS,
            default_value=0.5,
            min_value=0.0,
            max_value=1.0,
            category=ParameterCategory.RHYTHM
        )
    }

    # Add to registry
    for param_name, param_def in parameters.items():
        batch_gen.data_generator.registry.parameters[param_name] = param_def

    print(f"Generating data for {len(parameters)} parameters:")
    for param_name in parameters.keys():
        print(f"  - {param_name}")
    print()

    try:
        # Generate for all parameters
        results = batch_gen.generate_for_multiple_parameters(
            param_names=list(parameters.keys()),
            n_examples_per_param=50  # Small for example
        )

        print(f"\n✅ BATCH GENERATION COMPLETE")
        print(f"\nResults Summary:")
        for param_name, examples in results.items():
            if examples:
                avg_coherence = np.mean([ex.coherence_score for ex in examples])
                print(f"  {param_name}:")
                print(f"    Examples: {len(examples)}")
                print(f"    Avg Coherence: {avg_coherence:.3f}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXAMPLE 4: Custom Validation
# ============================================================================

def example_custom_validation():
    """
    Use custom validation criteria
    """
    print("\n" + "="*60)
    print("EXAMPLE 4: Custom Validation Criteria")
    print("="*60 + "\n")

    # Create custom validator with strict mode
    strict_validator = MusicalCoherenceValidator(strict_mode=True)

    # Initialize generator with strict validator
    generator = SyntheticTrainingDataGenerator(
        validator=strict_validator,
        output_root=Path('example_training_data_strict')
    )

    # Define parameter
    param_name = "example.dynamics.velocity_variance"
    param_def = ParameterDefinition(
        name="velocity_variance",
        full_path=param_name,
        description="Variance in note velocities",
        param_type=ParameterType.CONTINUOUS,
        default_value=20.0,
        min_value=0.0,
        max_value=40.0,
        category=ParameterCategory.DYNAMICS
    )

    print(f"Generating with STRICT validation for: {param_name}")
    print(f"Min coherence threshold: 0.7")
    print()

    try:
        training_data = generator.generate_training_data(
            param_name=param_name,
            param_def=param_def,
            n_examples=50,
            min_coherence=0.7  # Higher threshold
        )

        print(f"\n✅ SUCCESS: Generated {len(training_data)} high-quality examples")

        # Show validation details for first example
        if training_data:
            import mido
            first_example = training_data[0]
            midi = mido.MidiFile(str(first_example.midi_file))

            score, details = strict_validator.validate_with_details(midi)

            print(f"\nValidation Breakdown (first example):")
            print(f"  Overall Score: {score:.3f}")
            for aspect, value in details.items():
                print(f"    {aspect}: {value:.3f}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()


# ============================================================================
# EXAMPLE 5: Parameter Space Sampling Analysis
# ============================================================================

def example_sampling_analysis():
    """
    Analyze parameter space sampling quality
    """
    print("\n" + "="*60)
    print("EXAMPLE 5: Parameter Space Sampling Analysis")
    print("="*60 + "\n")

    sampler = ParameterSpaceSampler(seed=42)

    # Test continuous parameter sampling
    continuous_param = ParameterDefinition(
        name="test_continuous",
        full_path="test.continuous",
        description="Test continuous parameter",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0
    )

    print("Sampling 1000 values from continuous parameter [0.0, 1.0]")
    samples = sampler.sample_parameter_space(continuous_param, 1000)

    print(f"\nSampling Statistics:")
    print(f"  Mean: {np.mean(samples):.3f} (expected: 0.5)")
    print(f"  Std: {np.std(samples):.3f} (expected: ~0.29)")
    print(f"  Min: {np.min(samples):.3f}")
    print(f"  Max: {np.max(samples):.3f}")

    # Check coverage using histogram
    hist, edges = np.histogram(samples, bins=10)
    print(f"\nCoverage (10 bins):")
    for i, count in enumerate(hist):
        print(f"  [{edges[i]:.1f}, {edges[i+1]:.1f}]: {count} samples")

    # Test categorical parameter sampling
    categorical_param = ParameterDefinition(
        name="test_categorical",
        full_path="test.categorical",
        description="Test categorical parameter",
        param_type=ParameterType.CATEGORICAL,
        default_value="major",
        options=["major", "minor", "diminished", "augmented"]
    )

    print("\n" + "-"*60)
    print("Sampling 1000 values from categorical parameter")
    print(f"Options: {categorical_param.options}")

    samples = sampler.sample_parameter_space(categorical_param, 1000)

    from collections import Counter
    counts = Counter(samples)

    print(f"\nBalance:")
    for option, count in counts.items():
        percentage = (count / len(samples)) * 100
        print(f"  {option}: {count} samples ({percentage:.1f}%)")


# ============================================================================
# EXAMPLE 6: Integration with XGBoost
# ============================================================================

def example_xgboost_integration():
    """
    Demonstrate integration with XGBoost training
    """
    print("\n" + "="*60)
    print("EXAMPLE 6: Integration with XGBoost Training")
    print("="*60 + "\n")

    try:
        import xgboost as xgb
    except ImportError:
        print("❌ XGBoost not installed. Install with: pip install xgboost")
        return

    # Generate training data
    generator = SyntheticTrainingDataGenerator(
        output_root=Path('example_xgboost_data')
    )

    param_name = "example.test.parameter"
    param_def = ParameterDefinition(
        name="test_parameter",
        full_path=param_name,
        description="Test parameter for XGBoost",
        param_type=ParameterType.CONTINUOUS,
        default_value=0.5,
        min_value=0.0,
        max_value=1.0
    )

    print("Generating training data...")
    training_data = generator.generate_training_data(
        param_name=param_name,
        param_def=param_def,
        n_examples=200
    )

    print(f"Generated {len(training_data)} examples")

    # Prepare XGBoost dataset
    print("\nPreparing XGBoost dataset...")
    X = np.vstack([ex.features for ex in training_data])
    y = np.array([ex.parameter_value for ex in training_data])

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    # Split train/test
    split_idx = int(len(training_data) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Train XGBoost model
    print("\nTraining XGBoost model...")
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=50,
        evals=evals,
        verbose_eval=10
    )

    # Evaluate
    print("\nEvaluating model...")
    predictions = model.predict(dtest)
    mse = np.mean((predictions - y_test) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test))

    print(f"\nTest Set Performance:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Save model
    model_path = Path('example_xgboost_data') / 'model.json'
    model.save_model(str(model_path))
    print(f"\n✅ Model saved to: {model_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("SYNTHETIC TRAINING DATA GENERATOR - EXAMPLE USAGE")
    print("="*60)

    examples = [
        ("Basic Generation", example_basic_generation),
        ("Genre-Balanced Generation", example_genre_balanced_generation),
        ("Batch Generation", example_batch_generation),
        ("Custom Validation", example_custom_validation),
        ("Sampling Analysis", example_sampling_analysis),
        ("XGBoost Integration", example_xgboost_integration),
    ]

    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print(f"\n  0. Run all examples")
    print(f"  q. Quit")

    while True:
        choice = input("\nSelect example (0-6, q to quit): ").strip()

        if choice.lower() == 'q':
            break
        elif choice == '0':
            for name, func in examples:
                func()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            idx = int(choice) - 1
            examples[idx][1]()
        else:
            print("Invalid choice. Please try again.")

    print("\n" + "="*60)
    print("Examples complete!")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
