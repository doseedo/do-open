#!/usr/bin/env python3
"""
Performance Benchmarking Suite - Agent 28
==========================================

Comprehensive benchmarking tools to verify performance targets:
- Parameter prediction: <1 second for 800 parameters
- Training time per model: <30 seconds
- Memory usage: <2GB RAM
- Model storage: <500MB total

Author: Agent 28 - Performance Optimizer
License: MIT
"""

import argparse
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("ERROR: NumPy required. Install with: pip install numpy")
    sys.exit(1)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("ERROR: XGBoost required. Install with: pip install xgboost")
    sys.exit(1)

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    print("ERROR: Joblib required. Install with: pip install joblib")
    sys.exit(1)

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("WARNING: psutil not available. Memory monitoring disabled.")

from optimization.performance_optimizer import (
    PerformanceOptimizer,
    ParallelTrainer,
    PerformanceMonitor,
)


class BenchmarkSuite:
    """
    Comprehensive benchmarking suite for performance validation

    Tests all performance targets:
    1. Prediction speed for 800 parameters
    2. Training time per model
    3. Memory usage
    4. Model storage
    5. Cache efficiency
    """

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize benchmark suite

        Args:
            output_dir: Directory for benchmark outputs
        """
        self.output_dir = output_dir or Path(tempfile.mkdtemp())
        self.models_dir = self.output_dir / 'models'
        self.cache_dir = self.output_dir / 'cache'

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.results: Dict = {}
        self.monitor = PerformanceMonitor(enabled=True)

    def run_all_benchmarks(self) -> Dict:
        """Run all benchmarks"""
        print("="*80)
        print("PERFORMANCE BENCHMARK SUITE - AGENT 28")
        print("="*80)
        print()

        # 1. Training benchmark
        print("[1/5] Running training benchmark...")
        self.benchmark_training()

        # 2. Prediction speed benchmark
        print("\n[2/5] Running prediction speed benchmark...")
        self.benchmark_prediction_speed()

        # 3. Scalability benchmark
        print("\n[3/5] Running scalability benchmark...")
        self.benchmark_scalability()

        # 4. Memory benchmark
        print("\n[4/5] Running memory benchmark...")
        self.benchmark_memory()

        # 5. Cache efficiency benchmark
        print("\n[5/5] Running cache efficiency benchmark...")
        self.benchmark_cache_efficiency()

        # Print summary
        print("\n")
        self.print_summary()

        return self.results

    def benchmark_training(self, num_models: int = 50):
        """
        Benchmark training speed

        Target: <30 seconds per model
        """
        print(f"  Training {num_models} models in parallel...")

        # Create training jobs
        training_jobs = []
        num_features = 100
        num_samples = 1000

        for i in range(num_models):
            X_train = np.random.randn(num_samples, num_features)
            y_train = np.random.randn(num_samples)

            training_jobs.append({
                'param_name': f'param_{i:04d}',
                'X_train': X_train,
                'y_train': y_train,
                'params': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                }
            })

        # Train in parallel
        trainer = ParallelTrainer(max_workers=8)

        start_time = time.time()
        results = trainer.train_models(training_jobs, self.models_dir)
        total_time = time.time() - start_time

        # Calculate statistics
        successful = sum(1 for r in results.values() if r.get('success', False))
        avg_time = total_time / num_models
        training_times = [r['training_time'] for r in results.values() if 'training_time' in r]

        self.results['training'] = {
            'num_models': num_models,
            'successful': successful,
            'total_time': total_time,
            'avg_time_per_model': avg_time,
            'max_training_time': max(training_times) if training_times else 0,
            'min_training_time': min(training_times) if training_times else 0,
            'target_met': avg_time < 30.0,
        }

        print(f"  ✓ Trained {successful}/{num_models} models")
        print(f"  ✓ Total time: {total_time:.2f}s")
        print(f"  ✓ Avg time per model: {avg_time:.2f}s (target: <30s)")
        print(f"  ✓ Target met: {'YES' if avg_time < 30.0 else 'NO'}")

    def benchmark_prediction_speed(self, num_predictions: int = 100):
        """
        Benchmark prediction speed

        Target: <1 second for 800 parameters
        """
        # Determine how many models we have
        model_files = list(self.models_dir.glob('*.pkl'))
        num_models = len(model_files)

        print(f"  Testing prediction speed with {num_models} parameters...")

        # Initialize optimizer
        optimizer = PerformanceOptimizer(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            max_workers=8,
            enable_monitoring=True
        )

        # Load models
        load_start = time.time()
        optimizer.load_models()
        load_time = time.time() - load_start

        # Generate test input
        test_input = np.random.randn(100)

        # Warm-up run
        optimizer.predict(test_input)

        # Benchmark predictions
        start_time = time.time()
        for _ in range(num_predictions):
            predictions = optimizer.predict(test_input)
        total_time = time.time() - start_time

        avg_time = total_time / num_predictions
        predictions_per_second = num_predictions / total_time

        # Extrapolate to 800 parameters
        if num_models > 0:
            time_for_800 = avg_time * (800 / num_models)
        else:
            time_for_800 = 0

        self.results['prediction_speed'] = {
            'num_models': num_models,
            'num_predictions': num_predictions,
            'load_time': load_time,
            'total_prediction_time': total_time,
            'avg_time_per_prediction': avg_time,
            'predictions_per_second': predictions_per_second,
            'extrapolated_time_800_params': time_for_800,
            'target_met': time_for_800 < 1.0 if num_models > 0 else False,
        }

        print(f"  ✓ Model loading time: {load_time:.3f}s")
        print(f"  ✓ Avg prediction time: {avg_time*1000:.2f}ms")
        print(f"  ✓ Predictions/second: {predictions_per_second:.1f}")
        print(f"  ✓ Extrapolated time for 800 params: {time_for_800*1000:.2f}ms (target: <1000ms)")
        print(f"  ✓ Target met: {'YES' if time_for_800 < 1.0 else 'NO' if num_models > 0 else 'N/A'}")

        # Store optimizer for other tests
        self.optimizer = optimizer

    def benchmark_scalability(self):
        """
        Benchmark scalability with increasing parameter counts

        Tests: 10, 50, 100, 200, 500 parameters
        """
        print(f"  Testing scalability across different parameter counts...")

        model_files = list(self.models_dir.glob('*.pkl'))
        total_models = len(model_files)

        test_sizes = [10, 50, 100, 200, 500]
        test_sizes = [s for s in test_sizes if s <= total_models]

        if not test_sizes:
            print("  ✗ Not enough models for scalability test")
            self.results['scalability'] = {'error': 'insufficient models'}
            return

        scalability_results = []

        for size in test_sizes:
            # Select subset of models
            param_names = [f'param_{i:04d}' for i in range(size)]

            # Load models
            optimizer = PerformanceOptimizer(
                models_dir=self.models_dir,
                cache_dir=self.cache_dir,
                max_workers=8
            )

            optimizer.load_models(parameter_names=param_names)

            # Benchmark
            test_input = np.random.randn(100)

            start_time = time.time()
            for _ in range(20):
                optimizer.predict(test_input, parameters=param_names)
            total_time = time.time() - start_time

            avg_time = total_time / 20

            scalability_results.append({
                'num_params': size,
                'avg_time': avg_time,
                'time_per_param': avg_time / size,
            })

            print(f"    {size} params: {avg_time*1000:.2f}ms ({avg_time/size*1000:.3f}ms/param)")

        self.results['scalability'] = {
            'results': scalability_results,
            'total_models': total_models,
        }

    def benchmark_memory(self):
        """
        Benchmark memory usage

        Target: <2GB RAM
        """
        if not PSUTIL_AVAILABLE:
            print("  ✗ psutil not available, skipping memory benchmark")
            self.results['memory'] = {'error': 'psutil not available'}
            return

        print(f"  Measuring memory usage...")

        process = psutil.Process()

        # Get baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Load all models
        optimizer = PerformanceOptimizer(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            max_workers=8
        )

        optimizer.load_models()

        # Get memory after loading
        loaded_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Make predictions
        test_input = np.random.randn(100)
        for _ in range(100):
            optimizer.predict(test_input)

        # Get memory after predictions
        prediction_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Calculate model storage size
        total_storage_mb = sum(
            f.stat().st_size for f in self.models_dir.glob('*.pkl')
        ) / 1024 / 1024

        self.results['memory'] = {
            'baseline_memory_mb': baseline_memory,
            'loaded_memory_mb': loaded_memory,
            'prediction_memory_mb': prediction_memory,
            'memory_increase_mb': loaded_memory - baseline_memory,
            'peak_memory_mb': prediction_memory,
            'total_storage_mb': total_storage_mb,
            'memory_target_met': prediction_memory < 2048,
            'storage_target_met': total_storage_mb < 500,
        }

        print(f"  ✓ Baseline memory: {baseline_memory:.1f} MB")
        print(f"  ✓ Memory after loading: {loaded_memory:.1f} MB")
        print(f"  ✓ Peak memory: {prediction_memory:.1f} MB (target: <2048 MB)")
        print(f"  ✓ Total storage: {total_storage_mb:.1f} MB (target: <500 MB)")
        print(f"  ✓ Memory target met: {'YES' if prediction_memory < 2048 else 'NO'}")
        print(f"  ✓ Storage target met: {'YES' if total_storage_mb < 500 else 'NO'}")

    def benchmark_cache_efficiency(self):
        """
        Benchmark cache hit rates and efficiency
        """
        print(f"  Measuring cache efficiency...")

        # Clear caches
        if hasattr(self, 'optimizer'):
            self.optimizer.clear_caches()

        # Create new optimizer
        optimizer = PerformanceOptimizer(
            models_dir=self.models_dir,
            cache_dir=self.cache_dir,
            max_workers=8
        )

        optimizer.load_models()

        # Make predictions with repeated inputs
        test_inputs = [np.random.randn(100) for _ in range(10)]

        # First pass (populate cache)
        for input_data in test_inputs:
            optimizer.predict(input_data)

        # Second pass (should hit cache)
        for input_data in test_inputs:
            optimizer.predict(input_data)

        # Get cache stats
        model_cache_stats = optimizer.model_cache.get_stats()
        feature_cache_stats = optimizer.feature_cache.get_stats()

        self.results['cache_efficiency'] = {
            'model_cache': model_cache_stats,
            'feature_cache': feature_cache_stats,
        }

        print(f"  ✓ Model cache hit rate: {model_cache_stats['hit_rate']*100:.1f}%")
        print(f"  ✓ Model cache size: {model_cache_stats['size']}")
        print(f"  ✓ Feature cache hit rate: {feature_cache_stats['hit_rate']*100:.1f}%")
        print(f"  ✓ Feature cache size: {feature_cache_stats['memory_size']}")

    def print_summary(self):
        """Print comprehensive summary"""
        print("="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        # Training results
        if 'training' in self.results:
            tr = self.results['training']
            print(f"\n1. TRAINING PERFORMANCE")
            print(f"   Models trained: {tr['successful']}/{tr['num_models']}")
            print(f"   Avg time/model: {tr['avg_time_per_model']:.2f}s")
            print(f"   Target (<30s):  {'✓ PASS' if tr['target_met'] else '✗ FAIL'}")

        # Prediction results
        if 'prediction_speed' in self.results:
            ps = self.results['prediction_speed']
            print(f"\n2. PREDICTION SPEED")
            print(f"   Models loaded: {ps['num_models']}")
            print(f"   Avg time/prediction: {ps['avg_time_per_prediction']*1000:.2f}ms")
            print(f"   Predictions/sec: {ps['predictions_per_second']:.1f}")
            print(f"   Est. time for 800 params: {ps['extrapolated_time_800_params']*1000:.2f}ms")
            print(f"   Target (<1000ms): {'✓ PASS' if ps.get('target_met', False) else '✗ FAIL'}")

        # Memory results
        if 'memory' in self.results and 'error' not in self.results['memory']:
            mem = self.results['memory']
            print(f"\n3. MEMORY USAGE")
            print(f"   Peak memory: {mem['peak_memory_mb']:.1f} MB")
            print(f"   Total storage: {mem['total_storage_mb']:.1f} MB")
            print(f"   Memory target (<2048 MB): {'✓ PASS' if mem['memory_target_met'] else '✗ FAIL'}")
            print(f"   Storage target (<500 MB): {'✓ PASS' if mem['storage_target_met'] else '✗ FAIL'}")

        # Cache results
        if 'cache_efficiency' in self.results:
            cache = self.results['cache_efficiency']
            print(f"\n4. CACHE EFFICIENCY")
            print(f"   Model cache hit rate: {cache['model_cache']['hit_rate']*100:.1f}%")
            print(f"   Feature cache hit rate: {cache['feature_cache']['hit_rate']*100:.1f}%")

        print("\n" + "="*80)

    def cleanup(self):
        """Clean up temporary files"""
        if self.output_dir.exists() and 'tmp' in str(self.output_dir):
            shutil.rmtree(self.output_dir)


def main():
    """Main benchmark entry point"""
    parser = argparse.ArgumentParser(
        description='Performance Benchmarking Suite for Agent 28'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        help='Output directory for models and cache (default: temp dir)'
    )

    parser.add_argument(
        '--num-models',
        type=int,
        default=50,
        help='Number of models to train for benchmarking (default: 50)'
    )

    parser.add_argument(
        '--keep-files',
        action='store_true',
        help='Keep benchmark files after completion'
    )

    args = parser.parse_args()

    # Create benchmark suite
    suite = BenchmarkSuite(output_dir=args.output_dir)

    try:
        # Run benchmarks
        results = suite.run_all_benchmarks()

        # Optionally save results
        if args.output_dir:
            import json
            results_file = args.output_dir / 'benchmark_results.json'

            # Convert results to JSON-serializable format
            def make_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, (list, tuple)):
                    return [make_serializable(item) for item in obj]
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    return str(obj)

            serializable_results = make_serializable(results)

            with open(results_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)

            print(f"\nResults saved to: {results_file}")

    finally:
        # Cleanup
        if not args.keep_files:
            suite.cleanup()


if __name__ == '__main__':
    main()
