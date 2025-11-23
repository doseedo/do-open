#!/usr/bin/env python3
"""
AGENT 9: PERFORMANCE BENCHMARKING SCRIPT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Comprehensive performance benchmarking for the semantic evaluation system.
Measures training time, extraction time, memory usage, and throughput.

Usage:
    python benchmark_semantic_evaluation.py
    python benchmark_semantic_evaluation.py --dataset-size 1000
    python benchmark_semantic_evaluation.py --device cuda

Author: Agent 9 - Testing & Validation Specialist
"""

import argparse
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List
import psutil
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict

from midi_generator.learning.semantic_evaluation import SemanticEvaluator
from midi_generator.learning.semantic_encoder import SemanticFeatureEncoder
from midi_generator.learning.semantic_features import SemanticFeatureBank, SemanticFeature


@dataclass
class BenchmarkResults:
    """Container for benchmark results."""

    # Training Performance
    training_time_seconds: float
    training_throughput_samples_per_sec: float

    # Extraction Performance
    extraction_time_ms_mean: float
    extraction_time_ms_std: float
    extraction_time_ms_p50: float
    extraction_time_ms_p95: float
    extraction_time_ms_p99: float
    extraction_throughput_samples_per_sec: float

    # Batch Extraction Performance
    batch_extraction_time_ms_mean: float
    batch_extraction_throughput: float

    # Evaluation Performance
    evaluation_time_seconds: float

    # Memory Usage
    memory_usage_mb_mean: float
    memory_usage_mb_peak: float

    # Model Size
    model_parameters: int
    model_size_mb: float

    # Accuracy (if available)
    reconstruction_r2: float = 0.0
    overall_quality: float = 0.0

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def save(self, filepath: Path):
        """Save results to JSON."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def print_summary(self):
        """Print formatted summary."""
        print("\n" + "=" * 70)
        print("PERFORMANCE BENCHMARK RESULTS")
        print("=" * 70)

        print("\n⚡ TRAINING PERFORMANCE")
        print(f"  Training Time:        {self.training_time_seconds:.2f} seconds")
        print(f"  Throughput:           {self.training_throughput_samples_per_sec:.2f} samples/sec")

        print("\n🚀 EXTRACTION PERFORMANCE (Single Sample)")
        print(f"  Mean Time:            {self.extraction_time_ms_mean:.2f} ms")
        print(f"  Std Dev:              {self.extraction_time_ms_std:.2f} ms")
        print(f"  Median (p50):         {self.extraction_time_ms_p50:.2f} ms")
        print(f"  95th Percentile:      {self.extraction_time_ms_p95:.2f} ms")
        print(f"  99th Percentile:      {self.extraction_time_ms_p99:.2f} ms")
        print(f"  Throughput:           {self.extraction_throughput_samples_per_sec:.2f} samples/sec")

        print("\n📦 BATCH EXTRACTION PERFORMANCE")
        print(f"  Mean Time:            {self.batch_extraction_time_ms_mean:.2f} ms")
        print(f"  Throughput:           {self.batch_extraction_throughput:.2f} samples/sec")

        print("\n🔬 EVALUATION PERFORMANCE")
        print(f"  Evaluation Time:      {self.evaluation_time_seconds:.2f} seconds")

        print("\n💾 MEMORY USAGE")
        print(f"  Mean Usage:           {self.memory_usage_mb_mean:.2f} MB")
        print(f"  Peak Usage:           {self.memory_usage_mb_peak:.2f} MB")

        print("\n📊 MODEL SIZE")
        print(f"  Parameters:           {self.model_parameters:,}")
        print(f"  Model Size:           {self.model_size_mb:.2f} MB")

        if self.reconstruction_r2 > 0:
            print("\n✅ QUALITY METRICS")
            print(f"  Reconstruction R²:    {self.reconstruction_r2:.4f}")
            print(f"  Overall Quality:      {self.overall_quality:.4f}")

        print("\n" + "=" * 70)


class SemanticEvaluationBenchmark:
    """Comprehensive benchmarking suite for semantic evaluation."""

    def __init__(
        self,
        input_dim: int = 200,
        semantic_dim: int = 30,
        dataset_size: int = 1000,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize benchmark.

        Args:
            input_dim: Input feature dimension
            semantic_dim: Semantic feature dimension
            dataset_size: Size of test dataset
            device: Computation device
        """
        self.input_dim = input_dim
        self.semantic_dim = semantic_dim
        self.dataset_size = dataset_size
        self.device = device

        print(f"\n🎯 Initializing benchmark on {device}")
        print(f"   Input dim: {input_dim}, Semantic dim: {semantic_dim}")
        print(f"   Dataset size: {dataset_size}")

        # Create components
        self.encoder = SemanticFeatureEncoder(
            input_dim=input_dim,
            output_dim=semantic_dim
        ).to(device)

        self.test_dataset = self._create_test_dataset()
        self.feature_bank = self._create_feature_bank()

        self.evaluator = SemanticEvaluator(
            encoder=self.encoder,
            feature_bank=self.feature_bank,
            test_dataset=self.test_dataset,
            device=device
        )

    def _create_test_dataset(self):
        """Create test dataset."""
        class BenchmarkDataset:
            def __init__(self, size, input_dim, device):
                self.size = size
                self.input_dim = input_dim
                self.device = device
                # Pre-generate data for consistent benchmarking
                self.data = [
                    {'features': torch.randn(input_dim).to(device)}
                    for _ in range(size)
                ]

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return self.data[idx]

        return BenchmarkDataset(self.dataset_size, self.input_dim, self.device)

    def _create_feature_bank(self):
        """Create feature bank."""
        bank = SemanticFeatureBank()

        for i in range(self.semantic_dim):
            feature = SemanticFeature(
                index=i,
                name=f"benchmark_feature_{i}",
                description=f"Benchmark feature {i}",
            )
            feature.musical_validity_score = np.random.uniform(0.7, 0.9)
            bank.add_feature(feature)

        return bank

    def benchmark_training(self, epochs: int = 10) -> Dict:
        """Benchmark training performance."""
        print("\n⚡ Benchmarking training...")

        # Simple training loop
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        self.encoder.train()

        start_time = time.time()

        for epoch in range(epochs):
            epoch_loss = 0.0

            for i in range(min(100, len(self.test_dataset))):  # Limit for benchmark
                sample = self.test_dataset[i]
                features = sample['features'].unsqueeze(0)

                # Forward pass
                optimizer.zero_grad()
                semantic = self.encoder.encode(features)
                reconstructed = self.encoder.decode(semantic)

                # Loss
                loss = criterion(reconstructed, features)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        training_time = time.time() - start_time
        throughput = (epochs * 100) / training_time

        print(f"   Training time: {training_time:.2f}s")
        print(f"   Throughput: {throughput:.2f} samples/sec")

        return {
            'training_time_seconds': training_time,
            'training_throughput_samples_per_sec': throughput,
        }

    def benchmark_extraction(self, num_samples: int = 1000) -> Dict:
        """Benchmark single-sample extraction performance."""
        print(f"\n🚀 Benchmarking extraction ({num_samples} samples)...")

        self.encoder.eval()

        extraction_times = []

        with torch.no_grad():
            # Warm up
            for _ in range(10):
                sample = self.test_dataset[0]
                _ = self.encoder.encode(sample['features'].unsqueeze(0))

            # Benchmark
            for i in range(min(num_samples, len(self.test_dataset))):
                sample = self.test_dataset[i % len(self.test_dataset)]
                features = sample['features'].unsqueeze(0)

                start = time.time()
                _ = self.encoder.encode(features)
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = (time.time() - start) * 1000  # ms

                extraction_times.append(elapsed)

        times_array = np.array(extraction_times)

        throughput = 1000.0 / times_array.mean()  # samples per second

        results = {
            'extraction_time_ms_mean': float(times_array.mean()),
            'extraction_time_ms_std': float(times_array.std()),
            'extraction_time_ms_p50': float(np.percentile(times_array, 50)),
            'extraction_time_ms_p95': float(np.percentile(times_array, 95)),
            'extraction_time_ms_p99': float(np.percentile(times_array, 99)),
            'extraction_throughput_samples_per_sec': throughput,
        }

        print(f"   Mean: {results['extraction_time_ms_mean']:.2f} ms")
        print(f"   P95: {results['extraction_time_ms_p95']:.2f} ms")
        print(f"   Throughput: {throughput:.2f} samples/sec")

        return results

    def benchmark_batch_extraction(self, batch_sizes: List[int] = [1, 8, 16, 32]) -> Dict:
        """Benchmark batch extraction performance."""
        print(f"\n📦 Benchmarking batch extraction...")

        self.encoder.eval()

        batch_results = {}

        with torch.no_grad():
            for batch_size in batch_sizes:
                times = []

                for _ in range(100):
                    # Create batch
                    batch = torch.stack([
                        self.test_dataset[i % len(self.test_dataset)]['features']
                        for i in range(batch_size)
                    ])

                    start = time.time()
                    _ = self.encoder.encode(batch)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    elapsed = (time.time() - start) * 1000  # ms

                    times.append(elapsed)

                mean_time = np.mean(times)
                throughput = (batch_size * 1000.0) / mean_time

                batch_results[f'batch_{batch_size}'] = {
                    'mean_time_ms': mean_time,
                    'throughput': throughput,
                }

                print(f"   Batch {batch_size}: {mean_time:.2f} ms ({throughput:.2f} samples/sec)")

        # Use largest batch size for overall metrics
        best_batch = batch_results[f'batch_{max(batch_sizes)}']

        return {
            'batch_extraction_time_ms_mean': best_batch['mean_time_ms'],
            'batch_extraction_throughput': best_batch['throughput'],
            'batch_results': batch_results,
        }

    def benchmark_evaluation(self) -> Dict:
        """Benchmark full evaluation suite."""
        print("\n🔬 Benchmarking evaluation suite...")

        start_time = time.time()
        metrics = self.evaluator.evaluate_all(verbose=False)
        evaluation_time = time.time() - start_time

        print(f"   Evaluation time: {evaluation_time:.2f}s")

        return {
            'evaluation_time_seconds': evaluation_time,
            'reconstruction_r2': metrics.reconstruction_r2,
            'overall_quality': metrics.overall_quality_score,
        }

    def benchmark_memory(self) -> Dict:
        """Benchmark memory usage."""
        print("\n💾 Benchmarking memory usage...")

        memory_readings = []

        # Get process
        process = psutil.Process()

        # Baseline
        baseline_memory = process.memory_info().rss / (1024 ** 2)  # MB

        self.encoder.eval()

        with torch.no_grad():
            for i in range(100):
                sample = self.test_dataset[i % len(self.test_dataset)]
                _ = self.encoder.encode(sample['features'].unsqueeze(0))

                # Measure memory
                current_memory = process.memory_info().rss / (1024 ** 2)  # MB
                memory_readings.append(current_memory - baseline_memory)

        memory_array = np.array(memory_readings)

        # GPU memory if available
        gpu_memory = 0.0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB

        results = {
            'memory_usage_mb_mean': float(memory_array.mean()),
            'memory_usage_mb_peak': float(memory_array.max()),
            'gpu_memory_mb': gpu_memory,
        }

        print(f"   Mean: {results['memory_usage_mb_mean']:.2f} MB")
        print(f"   Peak: {results['memory_usage_mb_peak']:.2f} MB")
        if gpu_memory > 0:
            print(f"   GPU: {gpu_memory:.2f} MB")

        return results

    def benchmark_model_size(self) -> Dict:
        """Benchmark model size."""
        print("\n📊 Measuring model size...")

        # Count parameters
        total_params = sum(p.numel() for p in self.encoder.parameters())

        # Estimate model size
        param_size = sum(p.numel() * p.element_size() for p in self.encoder.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.encoder.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 ** 2)

        results = {
            'model_parameters': total_params,
            'model_size_mb': model_size_mb,
        }

        print(f"   Parameters: {total_params:,}")
        print(f"   Size: {model_size_mb:.2f} MB")

        return results

    def run_all_benchmarks(self) -> BenchmarkResults:
        """Run all benchmarks."""
        print("\n" + "=" * 70)
        print("RUNNING COMPREHENSIVE BENCHMARK SUITE")
        print("=" * 70)

        results = {}

        # 1. Training
        results.update(self.benchmark_training(epochs=5))

        # 2. Extraction
        results.update(self.benchmark_extraction(num_samples=1000))

        # 3. Batch extraction
        batch_results = self.benchmark_batch_extraction()
        results['batch_extraction_time_ms_mean'] = batch_results['batch_extraction_time_ms_mean']
        results['batch_extraction_throughput'] = batch_results['batch_extraction_throughput']

        # 4. Evaluation
        results.update(self.benchmark_evaluation())

        # 5. Memory
        results.update(self.benchmark_memory())

        # 6. Model size
        results.update(self.benchmark_model_size())

        return BenchmarkResults(**results)


def main():
    """Main benchmark script."""
    parser = argparse.ArgumentParser(
        description='Benchmark semantic evaluation performance'
    )
    parser.add_argument(
        '--dataset-size',
        type=int,
        default=1000,
        help='Test dataset size'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Computation device'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Run benchmarks
    benchmark = SemanticEvaluationBenchmark(
        dataset_size=args.dataset_size,
        device=args.device
    )

    results = benchmark.run_all_benchmarks()

    # Print summary
    results.print_summary()

    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results.save(output_dir / 'benchmark_results.json')

    print(f"\n✅ Results saved to {output_dir}")

    # Check against target benchmarks
    print("\n" + "=" * 70)
    print("TARGET BENCHMARK COMPARISON")
    print("=" * 70)

    targets = {
        'Extraction Time': (results.extraction_time_ms_mean, 100.0, 'ms', '<'),
        'Reconstruction R²': (results.reconstruction_r2, 0.95, '', '>'),
        'Overall Quality': (results.overall_quality, 0.80, '', '>'),
    }

    for name, (actual, target, unit, comparison) in targets.items():
        if comparison == '<':
            status = "✅ PASS" if actual < target else "❌ FAIL"
        else:
            status = "✅ PASS" if actual > target else "❌ FAIL"

        print(f"{name:20s}: {actual:.2f}{unit} (target: {comparison}{target:.2f}{unit}) {status}")

    print("=" * 70)


if __name__ == "__main__":
    main()
