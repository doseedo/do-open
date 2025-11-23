"""
Benchmark GPU vs CPU performance for discovery pipeline.

Tests:
1. Sparse coding (FISTA algorithm) - Expected 50-100x speedup
2. Transform application - Expected 50-100x speedup
3. End-to-end iteration - Expected 15-50x speedup

Author: Agent 8 - GPU Tensorization
"""

import torch
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.tensor_representation import TensorMIDICorpus
from core.tensor_transforms import TensorTransformLibrary
from discovery.gpu_sparse_coding import GPUSparseEncoder


def benchmark_sparse_coding(B=2000, T=1000, F=132, M=500):
    """
    Compare GPU vs CPU sparse coding speed.

    Args:
        B: Batch size (number of pieces)
        T: Time steps
        F: Features
        M: Number of transforms
    """
    print("\n" + "="*70)
    print("BENCHMARK 1: SPARSE CODING")
    print("="*70)
    print(f"Corpus: {B} pieces × {T} steps × {F} features")
    print(f"Transforms: {M}")
    print(f"Total size: {B*T*F*4/1e9:.2f} GB")
    print()

    # Generate random data
    print("Generating test data...")
    pieces = torch.randn(B, T, F)
    transforms = torch.randn(M, T, F)

    # CPU benchmark
    print("\n" + "-"*70)
    print("CPU SPARSE CODING")
    print("-"*70)
    pieces_cpu = pieces.to('cpu')
    transforms_cpu = transforms.to('cpu')
    encoder_cpu = GPUSparseEncoder(device='cpu', max_iters=50)

    print("Running...")
    start = time.time()
    encodings_cpu, metrics_cpu = encoder_cpu.encode_batch(pieces_cpu, transforms_cpu, verbose=False)
    cpu_time = time.time() - start

    print(f"Time: {cpu_time:.1f}s")
    print(f"Iterations: {metrics_cpu['iterations']}")
    print(f"Sparsity: {metrics_cpu['sparsity_mean']:.1f} active transforms/piece")

    # GPU benchmark
    if torch.cuda.is_available():
        print("\n" + "-"*70)
        print("GPU SPARSE CODING")
        print("-"*70)
        pieces_gpu = pieces.to('cuda')
        transforms_gpu = transforms.to('cuda')
        encoder_gpu = GPUSparseEncoder(device='cuda', max_iters=50)

        # Warmup
        print("Warmup...")
        _, _ = encoder_gpu.encode_batch(pieces_gpu[:10], transforms_gpu, verbose=False)
        torch.cuda.synchronize()

        print("Running...")
        start = time.time()
        encodings_gpu, metrics_gpu = encoder_gpu.encode_batch(pieces_gpu, transforms_gpu, verbose=False)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"Time: {gpu_time:.1f}s")
        print(f"Iterations: {metrics_gpu['iterations']}")
        print(f"Sparsity: {metrics_gpu['sparsity_mean']:.1f} active transforms/piece")

        print("\n" + "="*70)
        print(f"SPEEDUP: {cpu_time/gpu_time:.1f}x")
        print("="*70)

        return cpu_time, gpu_time
    else:
        print("\n[!] GPU not available")
        return cpu_time, None


def benchmark_transform_application(B=2000, T=1000, F=132, num_transforms=100):
    """
    Compare GPU vs CPU transform application speed.

    Args:
        B: Batch size
        T: Time steps
        F: Features
        num_transforms: Number of transforms to apply
    """
    print("\n" + "="*70)
    print("BENCHMARK 2: TRANSFORM APPLICATION")
    print("="*70)
    print(f"Corpus: {B} pieces × {T} steps × {F} features")
    print(f"Operations: {num_transforms} transforms")
    print()

    lib = TensorTransformLibrary()

    # Generate test data
    print("Generating test data...")
    batch = torch.randn(B, T, F)

    # CPU
    print("\n" + "-"*70)
    print("CPU TRANSFORM APPLICATION")
    print("-"*70)
    batch_cpu = batch.to('cpu')

    print("Running...")
    start = time.time()
    result_cpu = batch_cpu
    for _ in range(num_transforms):
        result_cpu = lib.transpose_semitone(result_cpu, 7)
    cpu_time = time.time() - start

    print(f"Time: {cpu_time:.1f}s")
    print(f"Throughput: {num_transforms/cpu_time:.1f} transforms/sec")

    # GPU
    if torch.cuda.is_available():
        print("\n" + "-"*70)
        print("GPU TRANSFORM APPLICATION")
        print("-"*70)
        batch_gpu = batch.to('cuda')

        # Warmup
        print("Warmup...")
        _ = lib.transpose_semitone(batch_gpu, 7)
        torch.cuda.synchronize()

        print("Running...")
        start = time.time()
        result_gpu = batch_gpu
        for _ in range(num_transforms):
            result_gpu = lib.transpose_semitone(result_gpu, 7)
        torch.cuda.synchronize()
        gpu_time = time.time() - start

        print(f"Time: {gpu_time:.1f}s")
        print(f"Throughput: {num_transforms/gpu_time:.1f} transforms/sec")

        print("\n" + "="*70)
        print(f"SPEEDUP: {cpu_time/gpu_time:.1f}x")
        print("="*70)

        return cpu_time, gpu_time
    else:
        print("\n[!] GPU not available")
        return cpu_time, None


def benchmark_memory_usage():
    """Benchmark memory usage for typical corpus."""
    print("\n" + "="*70)
    print("BENCHMARK 3: MEMORY USAGE")
    print("="*70)

    if not torch.cuda.is_available():
        print("[!] GPU not available")
        return

    from core.gpu_memory_manager import GPUMemoryManager

    mem_manager = GPUMemoryManager(device='cuda')

    # Test different corpus sizes
    test_sizes = [
        (2000, 500, "Full corpus (2000 pieces, 500 transforms)"),
        (1000, 500, "Medium corpus (1000 pieces, 500 transforms)"),
        (500, 500, "Small corpus (500 pieces, 500 transforms)"),
        (2000, 100, "Full corpus (2000 pieces, 100 transforms)"),
    ]

    for num_pieces, num_transforms, description in test_sizes:
        print(f"\n{description}:")
        breakdown = mem_manager.estimate_discovery_memory(
            num_pieces,
            num_transforms,
            max_time_steps=2000,
            num_features=132
        )

        if breakdown['fits_in_memory']:
            print(f"  ✓ Fits in memory ({breakdown['total_estimated_gb']:.1f} GB)")
        else:
            print(f"  ✗ May exceed memory ({breakdown['total_estimated_gb']:.1f} GB)")


def benchmark_end_to_end_iteration():
    """Benchmark full discovery iteration (if test corpus available)."""
    print("\n" + "="*70)
    print("BENCHMARK 4: END-TO-END ITERATION (Simulated)")
    print("="*70)

    if not torch.cuda.is_available():
        print("[!] GPU not available")
        return

    print("\nThis would test a full discovery iteration:")
    print("1. Load corpus to GPU (2000 files)")
    print("2. Sparse coding (FISTA, 100 iterations)")
    print("3. Test 10,000 composition candidates")
    print("4. Select top 50 new transforms")
    print()
    print("Expected times:")
    print("  CPU: ~8-12 hours per iteration")
    print("  GPU: ~10-30 minutes per iteration")
    print("  Speedup: 15-50x")
    print()
    print("To run actual test, prepare a test corpus and use:")
    print("  python scripts/test_gpu_discovery.py --corpus path/to/corpus")


def print_summary(results):
    """Print summary of all benchmarks."""
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)

    sparse_coding_cpu, sparse_coding_gpu = results.get('sparse_coding', (None, None))
    transform_cpu, transform_gpu = results.get('transform', (None, None))

    if sparse_coding_gpu:
        print(f"\nSparse Coding:")
        print(f"  CPU: {sparse_coding_cpu:.1f}s")
        print(f"  GPU: {sparse_coding_gpu:.1f}s")
        print(f"  Speedup: {sparse_coding_cpu/sparse_coding_gpu:.1f}x")

    if transform_gpu:
        print(f"\nTransform Application:")
        print(f"  CPU: {transform_cpu:.1f}s")
        print(f"  GPU: {transform_gpu:.1f}s")
        print(f"  Speedup: {transform_cpu/transform_gpu:.1f}x")

    if sparse_coding_gpu and transform_gpu:
        avg_speedup = ((sparse_coding_cpu/sparse_coding_gpu) + (transform_cpu/transform_gpu)) / 2
        print(f"\nAverage Speedup: {avg_speedup:.1f}x")
        print()
        print("Expected end-to-end speedup: 15-50x")
        print("  (includes CPU-only overhead for pattern mining, code generation)")

    print("="*70)


def main():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("GPU TENSORIZATION BENCHMARK SUITE")
    print("="*70)
    print("\nThis benchmark compares GPU vs CPU performance for:")
    print("1. Sparse coding (FISTA algorithm)")
    print("2. Transform application")
    print("3. Memory usage estimation")
    print("4. End-to-end iteration (simulated)")

    if not torch.cuda.is_available():
        print("\n[!] WARNING: CUDA not available!")
        print("    GPU benchmarks will be skipped.")
        print("    Install PyTorch with CUDA:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu118")
    else:
        props = torch.cuda.get_device_properties(0)
        print(f"\nGPU: {props.name}")
        print(f"Memory: {props.total_memory/1e9:.1f} GB")

    print("\nPress Enter to start benchmarks (this will take 2-5 minutes)...")
    input()

    results = {}

    # Run benchmarks
    try:
        cpu_time, gpu_time = benchmark_sparse_coding(B=2000, T=1000, F=132, M=500)
        results['sparse_coding'] = (cpu_time, gpu_time)
    except Exception as e:
        print(f"[!] Sparse coding benchmark failed: {e}")

    try:
        cpu_time, gpu_time = benchmark_transform_application(B=2000, T=1000, F=132, num_transforms=100)
        results['transform'] = (cpu_time, gpu_time)
    except Exception as e:
        print(f"[!] Transform application benchmark failed: {e}")

    try:
        benchmark_memory_usage()
    except Exception as e:
        print(f"[!] Memory benchmark failed: {e}")

    try:
        benchmark_end_to_end_iteration()
    except Exception as e:
        print(f"[!] End-to-end benchmark failed: {e}")

    # Print summary
    print_summary(results)

    print("\n" + "="*70)
    print("BENCHMARKS COMPLETE")
    print("="*70)
    print()


if __name__ == '__main__':
    main()
