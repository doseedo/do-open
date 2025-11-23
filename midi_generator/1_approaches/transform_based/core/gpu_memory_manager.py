"""
Memory management for large batch processing on A100.

A100 has 80GB HBM2e - need to be careful with memory usage to avoid OOM errors.

Memory breakdown for discovery:
- Corpus tensor: ~2-4 GB (2000 pieces × 2000 steps × 132 features × 4 bytes)
- Transform dict: ~500 MB (500 transforms)
- Encodings: ~4 MB (2000 × 500 × 4 bytes)
- Working memory: ~10-20 GB (intermediate computations)
- Available for chunks: ~50-60 GB

Author: Agent 8 - GPU Tensorization
"""

import torch
from typing import Optional, Dict, Tuple


class GPUMemoryManager:
    """
    Manage GPU memory for batch processing.

    Features:
    - Estimate tensor memory usage
    - Compute optimal chunk sizes
    - Monitor memory utilization
    - Auto-clear cache when needed
    """

    def __init__(self, device: str = 'cuda', reserve_gb: float = 10.0):
        """
        Args:
            device: GPU device ('cuda', 'cuda:0', etc.)
            reserve_gb: Reserve this much memory for system/PyTorch overhead
        """
        self.device = device
        self.reserve_gb = reserve_gb

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            self.total_memory = props.total_memory / 1e9  # Convert to GB
            self.available_memory = self.total_memory - reserve_gb

            print(f"\n{'='*70}")
            print("GPU MEMORY MANAGER")
            print(f"{'='*70}")
            print(f"GPU: {props.name}")
            print(f"Total memory: {self.total_memory:.1f} GB")
            print(f"Reserved for system: {reserve_gb:.1f} GB")
            print(f"Available for processing: {self.available_memory:.1f} GB")
            print(f"{'='*70}\n")
        else:
            print("WARNING: CUDA not available, GPU memory management disabled")
            self.total_memory = 0
            self.available_memory = 0

    def estimate_tensor_memory(self, shape: tuple, dtype=torch.float32) -> float:
        """
        Estimate memory in GB for a tensor.

        Args:
            shape: Tensor shape tuple (e.g., (2000, 2000, 132))
            dtype: Data type (default: float32 = 4 bytes)

        Returns:
            memory_gb: Estimated memory in GB
        """
        num_elements = 1
        for dim in shape:
            num_elements *= dim

        bytes_per_element = torch.tensor([], dtype=dtype).element_size()
        total_bytes = num_elements * bytes_per_element

        return total_bytes / 1e9

    def compute_optimal_chunk_size(
        self,
        total_items: int,
        item_shape: tuple,
        safety_factor: float = 0.7,
        dtype=torch.float32
    ) -> int:
        """
        Compute optimal chunk size to fit in available memory.

        Args:
            total_items: Total number of items to process
            item_shape: Shape of each item (e.g., (2000, 132) for one piece)
            safety_factor: Use this fraction of available memory (0.7 = 70%)
            dtype: Data type

        Returns:
            chunk_size: Optimal number of items to process at once
        """
        memory_per_item = self.estimate_tensor_memory(item_shape, dtype)
        available_for_chunks = self.available_memory * safety_factor

        chunk_size = int(available_for_chunks / memory_per_item)
        chunk_size = min(chunk_size, total_items)
        chunk_size = max(1, chunk_size)  # At least 1

        print(f"Optimal chunk size: {chunk_size:,} items")
        print(f"  Memory per item: {memory_per_item*1000:.1f} MB")
        print(f"  Memory per chunk: {chunk_size * memory_per_item:.2f} GB")
        print(f"  Available memory: {available_for_chunks:.2f} GB")

        return chunk_size

    def estimate_discovery_memory(
        self,
        num_pieces: int,
        num_transforms: int,
        max_time_steps: int = 2000,
        num_features: int = 133
    ) -> Dict[str, float]:
        """
        Estimate total memory usage for discovery pipeline.

        Args:
            num_pieces: Number of MIDI files (B)
            num_transforms: Number of transforms (M)
            max_time_steps: Time steps (T)
            num_features: Features (F) - default 133 (includes program number)

        Returns:
            memory_breakdown: Dict with memory estimates in GB
        """
        # Corpus tensor: (B, T, F)
        corpus_gb = self.estimate_tensor_memory((num_pieces, max_time_steps, num_features))

        # Transform dictionary: (M, T, F)
        dict_gb = self.estimate_tensor_memory((num_transforms, max_time_steps, num_features))

        # Sparse encodings: (B, M)
        encodings_gb = self.estimate_tensor_memory((num_pieces, num_transforms))

        # Working memory estimates:
        # - FISTA intermediate tensors: ~2x corpus
        # - Composition testing: ~1x corpus
        # - Gradients and reconstruction: ~2x corpus
        working_gb = corpus_gb * 5

        total_gb = corpus_gb + dict_gb + encodings_gb + working_gb

        breakdown = {
            'corpus_tensor_gb': corpus_gb,
            'transform_dict_gb': dict_gb,
            'sparse_encodings_gb': encodings_gb,
            'working_memory_gb': working_gb,
            'total_estimated_gb': total_gb,
            'num_pieces': num_pieces,
            'num_transforms': num_transforms,
            'fits_in_memory': total_gb < self.available_memory
        }

        self._print_memory_breakdown(breakdown)

        return breakdown

    def _print_memory_breakdown(self, breakdown: Dict):
        """Pretty print memory breakdown."""
        print(f"\n{'='*70}")
        print("MEMORY ESTIMATE")
        print(f"{'='*70}")
        print(f"Corpus ({breakdown['num_pieces']} pieces):")
        print(f"  {breakdown['corpus_tensor_gb']:.2f} GB")
        print(f"Transform dict ({breakdown['num_transforms']} transforms):")
        print(f"  {breakdown['transform_dict_gb']:.2f} GB")
        print(f"Sparse encodings:")
        print(f"  {breakdown['sparse_encodings_gb']:.2f} GB")
        print(f"Working memory (gradients, intermediates):")
        print(f"  {breakdown['working_memory_gb']:.2f} GB")
        print(f"{'-'*70}")
        print(f"TOTAL: {breakdown['total_estimated_gb']:.2f} GB")
        print(f"Available: {self.available_memory:.2f} GB")

        if breakdown['fits_in_memory']:
            print(f"✓ FITS IN MEMORY")
        else:
            print(f"✗ MAY EXCEED MEMORY - Use chunking!")
        print(f"{'='*70}\n")

    def recommend_batch_size(
        self,
        total_pieces: int,
        num_transforms: int = 500,
        max_time_steps: int = 2000,
        num_features: int = 133
    ) -> Tuple[int, str]:
        """
        Recommend batch size for processing.

        Args:
            total_pieces: Total pieces in corpus
            num_transforms: Number of transforms
            max_time_steps: Time steps
            num_features: Features (default 133)

        Returns:
            recommended_batch_size: Optimal batch size
            recommendation: Human-readable recommendation
        """
        # Test different batch sizes
        test_sizes = [2000, 1000, 500, 250, 100]

        for batch_size in test_sizes:
            breakdown = self.estimate_discovery_memory(
                batch_size,
                num_transforms,
                max_time_steps,
                num_features
            )

            if breakdown['fits_in_memory']:
                if batch_size >= total_pieces:
                    recommendation = f"Process all {total_pieces} pieces at once (fits in memory)"
                else:
                    num_batches = (total_pieces + batch_size - 1) // batch_size
                    recommendation = f"Process in {num_batches} batches of {batch_size} pieces"

                return batch_size, recommendation

        # If nothing fits, use smallest and warn
        batch_size = 100
        num_batches = (total_pieces + batch_size - 1) // batch_size
        recommendation = f"WARNING: Use {num_batches} batches of {batch_size} (memory constrained)"

        return batch_size, recommendation

    def clear_cache(self):
        """Clear GPU cache to free memory."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")

    def print_memory_stats(self):
        """Print current GPU memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1e9
            reserved = torch.cuda.memory_reserved(self.device) / 1e9
            free = self.total_memory - reserved

            print(f"\n{'='*70}")
            print("CURRENT GPU MEMORY")
            print(f"{'='*70}")
            print(f"Allocated: {allocated:.2f} GB ({allocated/self.total_memory*100:.1f}%)")
            print(f"Reserved:  {reserved:.2f} GB ({reserved/self.total_memory*100:.1f}%)")
            print(f"Free:      {free:.2f} GB ({free/self.total_memory*100:.1f}%)")
            print(f"{'='*70}\n")
        else:
            print("CUDA not available")

    def get_memory_summary(self) -> Dict[str, float]:
        """
        Get current memory usage summary.

        Returns:
            summary: Dict with memory stats in GB
        """
        if not torch.cuda.is_available():
            return {
                'total_gb': 0,
                'allocated_gb': 0,
                'reserved_gb': 0,
                'free_gb': 0
            }

        allocated = torch.cuda.memory_allocated(self.device) / 1e9
        reserved = torch.cuda.memory_reserved(self.device) / 1e9
        free = self.total_memory - reserved

        return {
            'total_gb': self.total_memory,
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'free_gb': free,
            'utilization_pct': (allocated / self.total_memory) * 100
        }

    def monitor_memory(self, operation_name: str):
        """
        Context manager to monitor memory usage during operation.

        Usage:
            with mem_manager.monitor_memory("Sparse coding"):
                # ... operation ...
        """
        return MemoryMonitor(self, operation_name)


class MemoryMonitor:
    """Context manager for memory monitoring."""

    def __init__(self, manager: GPUMemoryManager, operation_name: str):
        self.manager = manager
        self.operation_name = operation_name
        self.start_allocated = 0
        self.start_reserved = 0

    def __enter__(self):
        if torch.cuda.is_available():
            self.start_allocated = torch.cuda.memory_allocated(self.manager.device) / 1e9
            self.start_reserved = torch.cuda.memory_reserved(self.manager.device) / 1e9
            print(f"\n[{self.operation_name}] Starting...")
            print(f"  Memory before: {self.start_allocated:.2f} GB allocated")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            end_allocated = torch.cuda.memory_allocated(self.manager.device) / 1e9
            end_reserved = torch.cuda.memory_reserved(self.manager.device) / 1e9

            delta_allocated = end_allocated - self.start_allocated
            delta_reserved = end_reserved - self.start_reserved

            print(f"[{self.operation_name}] Complete")
            print(f"  Memory after: {end_allocated:.2f} GB allocated")
            print(f"  Delta: +{delta_allocated:.2f} GB allocated, +{delta_reserved:.2f} GB reserved\n")


# ============================================================================
# Utility Functions
# ============================================================================

def check_gpu_availability() -> Dict[str, any]:
    """
    Check if GPU is available and get specs.

    Returns:
        gpu_info: Dict with GPU information
    """
    if not torch.cuda.is_available():
        return {
            'available': False,
            'count': 0,
            'message': 'CUDA not available. Install PyTorch with CUDA support.'
        }

    device_count = torch.cuda.device_count()
    devices = []

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        devices.append({
            'id': i,
            'name': props.name,
            'total_memory_gb': props.total_memory / 1e9,
            'compute_capability': (props.major, props.minor)
        })

    return {
        'available': True,
        'count': device_count,
        'devices': devices,
        'cuda_version': torch.version.cuda,
        'pytorch_version': torch.__version__
    }


def print_gpu_info():
    """Print GPU information."""
    info = check_gpu_availability()

    print(f"\n{'='*70}")
    print("GPU INFORMATION")
    print(f"{'='*70}")

    if not info['available']:
        print(info['message'])
    else:
        print(f"CUDA available: Yes")
        print(f"CUDA version: {info['cuda_version']}")
        print(f"PyTorch version: {info['pytorch_version']}")
        print(f"GPU count: {info['count']}")
        print()

        for device in info['devices']:
            print(f"GPU {device['id']}: {device['name']}")
            print(f"  Memory: {device['total_memory_gb']:.1f} GB")
            print(f"  Compute capability: {device['compute_capability'][0]}.{device['compute_capability'][1]}")

    print(f"{'='*70}\n")
