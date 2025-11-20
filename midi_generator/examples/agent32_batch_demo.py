"""
AGENT 32: Batch Processing Manager - Demo
==========================================

Demonstrates efficient batch processing capabilities:
1. Parallel MIDI feature extraction
2. Batch parameter prediction
3. Parallel training data generation
4. Batch model training
5. Progress tracking and monitoring
6. Performance benchmarking

Author: Agent 32 - Batch Processing Manager
License: MIT
"""

import time
from pathlib import Path
import numpy as np

from midi_generator.processing import (
    BatchProcessingManager,
    ProcessingMode,
    create_batch_manager
)


def demo_basic_batch_processing():
    """Demo 1: Basic batch processing with progress tracking"""
    print("\n" + "=" * 80)
    print("DEMO 1: Basic Batch Processing")
    print("=" * 80)

    # Create batch manager
    manager = BatchProcessingManager(n_workers=4, show_progress=True)

    # Example: Process 1000 items
    items = list(range(1000))

    def square_number(x):
        """Example processing function"""
        time.sleep(0.001)  # Simulate work
        return x ** 2

    # Process in parallel
    result = manager.batch_process(
        items,
        square_number,
        description="Squaring numbers"
    )

    print(f"\n{result.summary()}")
    print(f"\nFirst 10 results: {result.results[:10]}")

    manager.shutdown()


def demo_feature_extraction():
    """Demo 2: Parallel MIDI feature extraction"""
    print("\n" + "=" * 80)
    print("DEMO 2: Parallel Feature Extraction")
    print("=" * 80)

    # Create batch manager
    manager = BatchProcessingManager(n_workers=8, show_progress=True)

    # Example MIDI files (mock)
    midi_files = [
        Path(f"midi_files/example_{i}.mid")
        for i in range(50)
    ]

    print(f"\nProcessing {len(midi_files)} MIDI files...")
    print("(Using mock feature extractor for demo)")

    # Mock feature extraction
    class MockFeatureExtractor:
        def extract(self, midi_file):
            """Mock feature extraction"""
            time.sleep(0.05)  # Simulate extraction time
            return np.random.randn(1000)  # Return 1000 features

    try:
        # Extract features in parallel
        features = manager.batch_extract_features(
            midi_files,
            feature_extractor=MockFeatureExtractor(),
            output_format='array'
        )

        print(f"\n✅ Extracted features: {features.shape}")

    except Exception as e:
        print(f"\n⚠️ Demo requires actual MIDI files: {e}")

    manager.shutdown()


def demo_parameter_prediction():
    """Demo 3: Batch parameter prediction"""
    print("\n" + "=" * 80)
    print("DEMO 3: Batch Parameter Prediction")
    print("=" * 80)

    # Create batch manager
    manager = BatchProcessingManager(n_workers=4, show_progress=True)

    # Mock feature matrix (100 samples, 1000 features)
    feature_matrix = np.random.randn(100, 1000)

    print(f"\nPredicting parameters for {feature_matrix.shape[0]} samples...")
    print("(Using mock models for demo)")

    try:
        # Predict parameters
        predictions = manager.batch_predict_parameters(
            feature_matrix,
            parameter_names=['harmony.progression.style', 'melody.contour.shape'],
            models_dir=Path('midi_generator/models/pretrained')
        )

        print(f"\n✅ Generated predictions for {len(predictions)} samples")
        if predictions:
            print(f"   Parameters per sample: {len(predictions[0])}")

    except Exception as e:
        print(f"\n⚠️ Demo requires trained models: {e}")

    manager.shutdown()


def demo_training_data_generation():
    """Demo 4: Parallel training data generation"""
    print("\n" + "=" * 80)
    print("DEMO 4: Parallel Training Data Generation")
    print("=" * 80)

    # Create batch manager
    manager = BatchProcessingManager(n_workers=4, show_progress=True)

    # Parameters to generate data for
    parameter_names = [
        'harmony.voicing.spread',
        'melody.contour.direction',
        'rhythm.complexity',
        'dynamics.range',
        'articulation.staccato_probability'
    ]

    print(f"\nGenerating training data for {len(parameter_names)} parameters...")
    print("(100 examples per parameter)")

    try:
        # Generate training data in parallel
        results = manager.batch_generate_training_data(
            parameter_names,
            n_examples_per_param=100,
            output_dir=Path('training_data_demo')
        )

        print(f"\n✅ Generated data for {len(results['results'])} parameters")
        print(f"   Success rate: {results['success_rate']:.1%}")

    except Exception as e:
        print(f"\n⚠️ Demo error: {e}")

    manager.shutdown()


def demo_model_training():
    """Demo 5: Batch model training"""
    print("\n" + "=" * 80)
    print("DEMO 5: Batch Model Training")
    print("=" * 80)

    # Create batch manager (fewer workers for memory-intensive tasks)
    manager = BatchProcessingManager(n_workers=2, show_progress=True)

    # Parameters to train models for
    parameter_names = [
        'harmony.voicing.spread',
        'melody.contour.direction',
        'rhythm.complexity'
    ]

    print(f"\nTraining models for {len(parameter_names)} parameters...")

    try:
        # Train models in parallel
        results = manager.batch_train_models(
            parameter_names,
            training_data_dir=Path('training_data'),
            models_dir=Path('models_demo'),
            enable_tuning=False
        )

        print(f"\n✅ Trained {len(results['results'])} models")
        print(f"   Success rate: {results['success_rate']:.1%}")

    except Exception as e:
        print(f"\n⚠️ Demo requires training data: {e}")

    manager.shutdown()


def demo_performance_comparison():
    """Demo 6: Sequential vs Parallel performance comparison"""
    print("\n" + "=" * 80)
    print("DEMO 6: Performance Comparison")
    print("=" * 80)

    items = list(range(1000))

    def expensive_operation(x):
        """Simulate expensive computation"""
        time.sleep(0.01)
        result = 0
        for i in range(100):
            result += x * i
        return result

    # Sequential processing
    print("\n1. Sequential Processing:")
    manager_seq = BatchProcessingManager(
        n_workers=1,
        mode=ProcessingMode.SEQUENTIAL,
        show_progress=True
    )

    start = time.time()
    result_seq = manager_seq.batch_process(
        items,
        expensive_operation,
        description="Sequential"
    )
    time_seq = time.time() - start

    print(f"   Time: {time_seq:.2f}s")
    print(f"   Throughput: {result_seq.throughput:.2f} items/sec")

    manager_seq.shutdown()

    # Parallel processing (4 workers)
    print("\n2. Parallel Processing (4 workers):")
    manager_par = BatchProcessingManager(
        n_workers=4,
        mode=ProcessingMode.MULTIPROCESS,
        show_progress=True
    )

    start = time.time()
    result_par = manager_par.batch_process(
        items,
        expensive_operation,
        description="Parallel (4 workers)"
    )
    time_par = time.time() - start

    print(f"   Time: {time_par:.2f}s")
    print(f"   Throughput: {result_par.throughput:.2f} items/sec")

    manager_par.shutdown()

    # Parallel processing (8 workers)
    print("\n3. Parallel Processing (8 workers):")
    manager_par8 = BatchProcessingManager(
        n_workers=8,
        mode=ProcessingMode.MULTIPROCESS,
        show_progress=True
    )

    start = time.time()
    result_par8 = manager_par8.batch_process(
        items,
        expensive_operation,
        description="Parallel (8 workers)"
    )
    time_par8 = time.time() - start

    print(f"   Time: {time_par8:.2f}s")
    print(f"   Throughput: {result_par8.throughput:.2f} items/sec")

    manager_par8.shutdown()

    # Summary
    print(f"\n{'='*80}")
    print("PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    print(f"Sequential:          {time_seq:.2f}s (1.0x baseline)")
    print(f"Parallel (4 workers): {time_par:.2f}s ({time_seq/time_par:.2f}x speedup)")
    print(f"Parallel (8 workers): {time_par8:.2f}s ({time_seq/time_par8:.2f}x speedup)")
    print(f"{'='*80}")


def demo_error_handling():
    """Demo 7: Error handling and retry logic"""
    print("\n" + "=" * 80)
    print("DEMO 7: Error Handling")
    print("=" * 80)

    manager = BatchProcessingManager(
        n_workers=4,
        show_progress=True,
        retry_attempts=3
    )

    items = list(range(100))

    def unreliable_operation(x):
        """Operation that sometimes fails"""
        if x % 10 == 0:  # Fail on every 10th item
            raise ValueError(f"Simulated error for item {x}")
        return x * 2

    result = manager.batch_process(
        items,
        unreliable_operation,
        description="Processing with errors",
        collect_errors=True
    )

    print(f"\n{result.summary()}")

    if result.errors:
        print(f"\nError Details (first 5):")
        for idx, error in list(result.errors.items())[:5]:
            print(f"  Item {idx}: {error}")

    manager.shutdown()


def demo_statistics():
    """Demo 8: Statistics and monitoring"""
    print("\n" + "=" * 80)
    print("DEMO 8: Statistics and Monitoring")
    print("=" * 80)

    manager = BatchProcessingManager(n_workers=4, show_progress=True)

    # Run multiple operations
    print("\nRunning multiple batch operations...")

    for i in range(3):
        items = list(range(100 * (i + 1)))

        result = manager.batch_process(
            items,
            lambda x: x * 2,
            description=f"Operation {i+1}"
        )

        print(f"\nOperation {i+1} complete: {result.successful_items} items processed")

    # Print cumulative statistics
    manager.print_statistics()

    manager.shutdown()


def demo_context_manager():
    """Demo 9: Using context manager"""
    print("\n" + "=" * 80)
    print("DEMO 9: Context Manager Usage")
    print("=" * 80)

    items = list(range(500))

    # Use context manager for automatic cleanup
    with BatchProcessingManager(n_workers=4, show_progress=True) as manager:
        result = manager.batch_process(
            items,
            lambda x: x ** 2,
            description="Processing with context manager"
        )

        print(f"\n{result.summary()}")

    print("\n✅ Manager automatically cleaned up")


def demo_utility_function():
    """Demo 10: Using utility functions"""
    print("\n" + "=" * 80)
    print("DEMO 10: Utility Functions")
    print("=" * 80)

    # Create manager with recommended settings
    manager = create_batch_manager(n_workers=4, use_multiprocess=True)

    print(f"\n✅ Created manager with recommended settings:")
    print(f"   Workers: {manager.n_workers}")
    print(f"   Mode: {manager.mode.value}")
    print(f"   Chunk size: {manager.chunk_size}")

    items = list(range(200))

    result = manager.batch_process(
        items,
        lambda x: x * 3,
        description="Quick processing"
    )

    print(f"\n{result.summary()}")

    manager.shutdown()


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("AGENT 32: Batch Processing Manager - Demo Suite")
    print("=" * 80)
    print("\nThis demo showcases:")
    print("1. Basic batch processing")
    print("2. Parallel feature extraction")
    print("3. Batch parameter prediction")
    print("4. Parallel training data generation")
    print("5. Batch model training")
    print("6. Performance comparison")
    print("7. Error handling")
    print("8. Statistics and monitoring")
    print("9. Context manager usage")
    print("10. Utility functions")

    demos = [
        ("Basic Batch Processing", demo_basic_batch_processing),
        ("Feature Extraction", demo_feature_extraction),
        ("Parameter Prediction", demo_parameter_prediction),
        ("Training Data Generation", demo_training_data_generation),
        ("Model Training", demo_model_training),
        ("Performance Comparison", demo_performance_comparison),
        ("Error Handling", demo_error_handling),
        ("Statistics", demo_statistics),
        ("Context Manager", demo_context_manager),
        ("Utility Functions", demo_utility_function),
    ]

    print("\n" + "=" * 80)
    print("Select demo to run:")
    print("=" * 80)
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print("0. Run all demos")
    print("=" * 80)

    try:
        choice = input("\nEnter choice (0-10): ").strip()

        if choice == '0':
            for name, demo_fn in demos:
                try:
                    demo_fn()
                except KeyboardInterrupt:
                    print("\n\n⚠️ Demo interrupted by user")
                    break
                except Exception as e:
                    print(f"\n⚠️ Demo error: {e}")
                    import traceback
                    traceback.print_exc()

        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            name, demo_fn = demos[int(choice) - 1]
            demo_fn()

        else:
            print("Invalid choice")

    except KeyboardInterrupt:
        print("\n\n⚠️ Demo interrupted by user")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == '__main__':
    # Run a few quick demos
    demo_basic_batch_processing()
    demo_performance_comparison()
    demo_error_handling()
    demo_context_manager()

    print("\n\n" + "=" * 80)
    print("Quick demo complete!")
    print("Run with python -m midi_generator.examples.agent32_batch_demo for full menu")
    print("=" * 80)
