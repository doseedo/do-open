#!/usr/bin/env python3
"""
Performance Optimizer Integration Examples - Agent 28
=====================================================

Demonstrates how to integrate PerformanceOptimizer with the
Musical Program Synthesis system for optimal performance at scale.

Examples:
1. Basic prediction optimization
2. Batch MIDI analysis and parameter prediction
3. Training new parameter models in parallel
4. Model compression for deployment
5. Production deployment with monitoring

Author: Agent 28 - Performance Optimizer
License: MIT
"""

import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy required. Install with: pip install numpy")
    sys.exit(1)

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost required. Install with: pip install xgboost")
    sys.exit(1)

from optimization.performance_optimizer import (
    PerformanceOptimizer,
    ParallelTrainer,
    PerformanceMonitor,
    LRUModelCache,
    FeatureCache,
)


# ============================================================================
# Example 1: Basic Prediction Optimization
# ============================================================================

def example_1_basic_prediction():
    """
    Example 1: Basic prediction with caching and parallel processing
    """
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Prediction Optimization")
    print("="*80)

    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_model_cache_size=100,      # Cache up to 100 models
        max_model_memory_mb=1024,       # Max 1GB for model cache
        max_feature_memory_mb=512,      # Max 512MB for feature cache
        max_workers=8,                  # 8 parallel workers
        enable_monitoring=True          # Enable performance monitoring
    )

    # Load models for specific parameters
    print("\n1. Loading models...")
    parameter_names = [
        'harmony.voicing.type',
        'harmony.voicing.spread',
        'melody.contour',
        'rhythm.swing_ratio',
        'bass.pattern_complexity',
    ]

    models = optimizer.load_models(parameter_names=parameter_names)
    print(f"   Loaded {len(models)} models")

    # Prepare input data (e.g., features from MIDI analysis)
    print("\n2. Preparing input features...")
    # In real usage, this would come from deep_feature_extractor.py
    input_features = np.random.randn(100)  # 100 features

    # Make predictions
    print("\n3. Making predictions...")
    predictions = optimizer.predict(input_features)

    print(f"   Predicted {len(predictions)} parameters:")
    for param, value in predictions.items():
        print(f"     {param}: {value}")

    # Get performance stats
    print("\n4. Performance statistics:")
    stats = optimizer.get_stats()
    print(f"   Models loaded: {stats['num_loaded_models']}")
    print(f"   Model cache hit rate: {stats['model_cache']['hit_rate']*100:.1f}%")
    print(f"   Feature cache hit rate: {stats['feature_cache']['hit_rate']*100:.1f}%")

    return optimizer


# ============================================================================
# Example 2: Batch MIDI Analysis
# ============================================================================

def example_2_batch_analysis():
    """
    Example 2: Analyze multiple MIDI files in batch
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Batch MIDI Analysis")
    print("="*80)

    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_workers=8,
        enable_monitoring=True
    )

    # Load all available models
    print("\n1. Loading all models...")
    models = optimizer.load_models()
    print(f"   Loaded {len(models)} models")

    # Simulate batch of MIDI files
    print("\n2. Processing batch of MIDI files...")
    num_files = 20

    # In real usage, you would:
    # 1. Read MIDI files
    # 2. Extract features using deep_feature_extractor.py
    # 3. Pass features to optimizer

    # For this example, generate dummy features
    midi_features_batch = [
        np.random.randn(100) for _ in range(num_files)
    ]

    # Process batch
    start_time = time.time()
    results = optimizer.predict_batch(midi_features_batch)
    total_time = time.time() - start_time

    print(f"\n3. Results:")
    print(f"   Processed {len(results)} files in {total_time:.2f}s")
    print(f"   Average time per file: {total_time/len(results):.3f}s")
    print(f"   Files per second: {len(results)/total_time:.1f}")

    # Show example results
    print(f"\n4. Example predictions from first file:")
    first_result = results[0]
    for i, (param, value) in enumerate(first_result.items()):
        if i >= 5:  # Show first 5
            print(f"      ... and {len(first_result)-5} more")
            break
        print(f"     {param}: {value}")

    return optimizer


# ============================================================================
# Example 3: Parallel Training
# ============================================================================

def example_3_parallel_training():
    """
    Example 3: Train multiple parameter models in parallel
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Parallel Model Training")
    print("="*80)

    # Create trainer
    trainer = ParallelTrainer(max_workers=8)

    # Prepare training data
    print("\n1. Preparing training data...")

    # In real usage, training data would come from:
    # 1. MIDI files analyzed with deep_feature_extractor.py
    # 2. Known parameter values (ground truth)
    # 3. Synthetic training data from generator

    num_parameters = 20
    num_samples = 1000
    num_features = 100

    training_jobs = []

    for i in range(num_parameters):
        # Generate synthetic training data
        X_train = np.random.randn(num_samples, num_features)
        y_train = np.random.randn(num_samples)

        training_jobs.append({
            'param_name': f'param_{i:03d}',
            'X_train': X_train,
            'y_train': y_train,
            'params': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
            }
        })

    print(f"   Created {len(training_jobs)} training jobs")

    # Train models in parallel
    print(f"\n2. Training {num_parameters} models in parallel...")

    output_dir = Path('models_trained')
    output_dir.mkdir(exist_ok=True)

    # Progress callback
    def progress_callback(completed, total, param_name):
        print(f"   [{completed}/{total}] Trained {param_name}")

    start_time = time.time()
    results = trainer.train_models(
        training_jobs,
        output_dir,
        progress_callback=progress_callback
    )
    total_time = time.time() - start_time

    # Show results
    successful = sum(1 for r in results.values() if r.get('success', False))

    print(f"\n3. Training Results:")
    print(f"   Successfully trained: {successful}/{num_parameters}")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Average time per model: {total_time/num_parameters:.2f}s")

    # Show individual results
    print(f"\n4. Individual model results:")
    for param_name, result in list(results.items())[:5]:
        if result.get('success'):
            print(f"     {param_name}: {result['training_time']:.2f}s ({result['num_samples']} samples)")

    return results


# ============================================================================
# Example 4: Model Compression
# ============================================================================

def example_4_model_compression():
    """
    Example 4: Compress models for deployment
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Model Compression")
    print("="*80)

    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
    )

    print("\n1. Compressing models...")

    # Compress all models in directory
    compression_ratios = optimizer.compress_models(
        input_dir=Path('models'),
        output_dir=Path('models_compressed'),
        method='both',  # prune + quantize
        importance_threshold=0.01,
        tree_threshold=0.1
    )

    print(f"\n2. Compression Results:")
    print(f"   Compressed {len(compression_ratios)} models")

    if compression_ratios:
        avg_ratio = np.mean(list(compression_ratios.values()))
        print(f"   Average compression ratio: {avg_ratio:.2f}x")
        print(f"\n3. Top compressed models:")
        for model_name, ratio in sorted(compression_ratios.items(),
                                       key=lambda x: x[1],
                                       reverse=True)[:5]:
            print(f"     {model_name}: {ratio:.2f}x")

    return compression_ratios


# ============================================================================
# Example 5: Production Deployment with Monitoring
# ============================================================================

def example_5_production_deployment():
    """
    Example 5: Production deployment with comprehensive monitoring
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Production Deployment")
    print("="*80)

    # Initialize optimizer with production settings
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_model_cache_size=200,       # Larger cache for production
        max_model_memory_mb=2048,       # 2GB for models
        max_feature_memory_mb=1024,     # 1GB for features
        max_workers=16,                 # More workers for production
        enable_monitoring=True          # Critical for production
    )

    # Load models with priority
    print("\n1. Loading models with priority...")

    # Define priority: higher = more frequently used
    priority = {
        'harmony.voicing.type': 100,
        'melody.contour': 90,
        'rhythm.swing_ratio': 85,
        'bass.pattern_complexity': 75,
        # ... more parameters
    }

    # Load high-priority models first
    models = optimizer.load_models(priority=priority)
    print(f"   Loaded {len(models)} models")

    # Simulate production workload
    print("\n2. Running production workload...")

    num_requests = 100
    start_time = time.time()

    for i in range(num_requests):
        # Generate random input (in production, from MIDI analysis)
        input_features = np.random.randn(100)

        # Make prediction
        predictions = optimizer.predict(input_features)

        # Simulate request delay
        time.sleep(0.01)

    total_time = time.time() - start_time

    print(f"\n3. Workload Results:")
    print(f"   Processed {num_requests} requests")
    print(f"   Total time: {total_time:.2f}s")
    print(f"   Requests per second: {num_requests/total_time:.1f}")
    print(f"   Average latency: {total_time/num_requests*1000:.2f}ms")

    # Print comprehensive performance report
    print("\n4. Performance Report:")
    optimizer.print_report()

    # Run benchmark
    print("\n5. Running benchmark...")
    benchmark_results = optimizer.benchmark(num_predictions=50)

    print(f"   Benchmark results:")
    print(f"   - Avg prediction time: {benchmark_results['avg_time_per_prediction']*1000:.2f}ms")
    print(f"   - Predictions/sec: {benchmark_results['predictions_per_second']:.1f}")

    return optimizer


# ============================================================================
# Example 6: Integration with Feature Extractor
# ============================================================================

def example_6_feature_extractor_integration():
    """
    Example 6: Full integration with feature extraction
    """
    print("\n" + "="*80)
    print("EXAMPLE 6: Feature Extractor Integration")
    print("="*80)

    # Define feature extraction function
    # In real usage, this would call deep_feature_extractor.py
    def extract_features_from_midi(midi_path):
        """
        Extract features from MIDI file

        In production, this would:
        1. Load MIDI file
        2. Call deep_feature_extractor.extract_features()
        3. Return 1000+ features
        """
        # For this example, return dummy features
        return np.random.randn(1000)

    # Initialize optimizer
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_workers=8,
        enable_monitoring=True
    )

    # Load models
    print("\n1. Loading models...")
    models = optimizer.load_models()
    print(f"   Loaded {len(models)} models")

    # Process MIDI file
    print("\n2. Processing MIDI file...")

    midi_path = Path('example.mid')  # Placeholder

    # Extract features (with caching)
    features = extract_features_from_midi(midi_path)
    print(f"   Extracted {len(features)} features")

    # Predict parameters
    predictions = optimizer.predict(
        features,
        feature_extractor=None  # Features already extracted
    )

    print(f"\n3. Predicted parameters:")
    for i, (param, value) in enumerate(predictions.items()):
        if i >= 10:  # Show first 10
            print(f"     ... and {len(predictions)-10} more")
            break
        print(f"     {param}: {value}")

    # Process batch of MIDI files
    print("\n4. Processing multiple MIDI files...")

    midi_paths = [Path(f'example_{i}.mid') for i in range(10)]

    # Extract features for all files
    features_batch = [
        extract_features_from_midi(path)
        for path in midi_paths
    ]

    # Predict in batch
    results_batch = optimizer.predict_batch(features_batch)

    print(f"   Processed {len(results_batch)} files")

    return optimizer


# ============================================================================
# Example 7: Adaptive Optimization
# ============================================================================

def example_7_adaptive_optimization():
    """
    Example 7: Adaptive optimization based on workload
    """
    print("\n" + "="*80)
    print("EXAMPLE 7: Adaptive Optimization")
    print("="*80)

    # Start with conservative settings
    optimizer = PerformanceOptimizer(
        models_dir=Path('models'),
        cache_dir=Path('cache'),
        max_model_cache_size=50,
        max_workers=4,
        enable_monitoring=True
    )

    # Load models
    print("\n1. Initial configuration:")
    print(f"   Cache size: {optimizer.model_cache.max_size}")
    print(f"   Workers: {optimizer.max_workers}")

    models = optimizer.load_models()

    # Simulate workload and analyze
    print("\n2. Running initial workload...")

    for _ in range(50):
        input_data = np.random.randn(100)
        optimizer.predict(input_data)

    # Analyze cache performance
    cache_stats = optimizer.model_cache.get_stats()
    hit_rate = cache_stats['hit_rate']

    print(f"   Cache hit rate: {hit_rate*100:.1f}%")

    # Adapt configuration based on performance
    if hit_rate < 0.8:
        print("\n3. Cache hit rate low - increasing cache size...")
        optimizer.model_cache.max_size = 100
        print(f"   New cache size: {optimizer.model_cache.max_size}")

    # Run another workload
    print("\n4. Running optimized workload...")

    for _ in range(50):
        input_data = np.random.randn(100)
        optimizer.predict(input_data)

    # Check improvement
    new_stats = optimizer.model_cache.get_stats()
    new_hit_rate = new_stats['hit_rate']

    print(f"\n5. Results:")
    print(f"   Initial hit rate: {hit_rate*100:.1f}%")
    print(f"   Optimized hit rate: {new_hit_rate*100:.1f}%")
    print(f"   Improvement: {(new_hit_rate - hit_rate)*100:.1f}%")

    return optimizer


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("PERFORMANCE OPTIMIZER INTEGRATION EXAMPLES")
    print("Agent 28 - Performance Optimizer")
    print("="*80)

    # Note: In production, you would run one example at a time
    # Here we just demonstrate the usage patterns

    examples = [
        ("Basic Prediction", example_1_basic_prediction),
        ("Batch Analysis", example_2_batch_analysis),
        ("Parallel Training", example_3_parallel_training),
        ("Model Compression", example_4_model_compression),
        ("Production Deployment", example_5_production_deployment),
        ("Feature Extractor Integration", example_6_feature_extractor_integration),
        ("Adaptive Optimization", example_7_adaptive_optimization),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nNote: These are demonstration examples.")
    print("In production, you would run them individually with real data.")

    # You can run individual examples by uncommenting:
    # example_1_basic_prediction()
    # example_2_batch_analysis()
    # etc.


if __name__ == '__main__':
    main()
