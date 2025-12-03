"""
FAISS-accelerated cross-piece MDL optimization.

Uses approximate nearest neighbor search to find derivations across all pieces
in ~30 minutes on A100 instead of days/weeks with brute force.

Key innovation: Test inverse transforms with FAISS search
    Instead of: for each source, transform and compare to target
    Do: inverse-transform target, search for matching source in FAISS

Memory efficient: Processes 50K objects at a time (~17GB)

Author: Agent - Cross-Piece FAISS Implementation
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Set, Optional
from dataclasses import dataclass
from collections import defaultdict

# Try to import CuPy for GPU-resident comparisons
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


@dataclass
class FAISSSearchResult:
    """Result from FAISS approximate nearest neighbor search."""
    target_idx: int
    source_idx: int
    transform_idx: int
    error: float
    distance: float


@dataclass
class CrossPieceMDLStats:
    """Statistics from cross-piece MDL iteration."""
    iteration: int
    new_derivations: int
    paths_shortened: int
    total_sources: int
    avg_path_length: float
    total_description_length: int
    time_seconds: float


class HierarchicalGPUIndex:
    """
    GPU-optimized hierarchical index for large scales.

    Splits large scales (>15K objects) into GPU-safe chunks of 15K each,
    distributes across multiple GPUs, and merges search results.

    This avoids CUBLAS batch matrix multiply crashes while maximizing GPU utilization.

    Example:
        Scale 16: 60,507 objects → 4 chunks of ~15K each
        - Chunk 0 (15K) → GPU 0
        - Chunk 1 (15K) → GPU 1
        - Chunk 2 (15K) → GPU 0
        - Chunk 3 (15K) → GPU 1
        Search all 4 in parallel, merge top-k results = 3-4 min (vs 18 min on CPU)
    """

    # GPU-safe chunk size: 5K objects to avoid CUBLAS batch matrix multiply limits
    # Error with 15K was: (512, 2128) x (15000, 2128) - too large for CUBLAS
    # 5K keeps each chunk small enough even with large query batches and high dims
    MAX_GPU_SAFE_CHUNK = 5000

    # Max query batch size for GPU search (to avoid CUBLAS crashes)
    MAX_QUERY_BATCH = 256

    def __init__(self, sub_indices: List, offsets: List[int], gpu_indices: List = None):
        """
        Args:
            sub_indices: List of FAISS indices (CPU)
            offsets: List of starting indices for each chunk (for global index mapping)
            gpu_indices: Optional list of GPU-transferred indices
        """
        self.sub_indices = sub_indices
        self.offsets = offsets
        self.gpu_indices = gpu_indices or []
        self.ntotal = sum(idx.ntotal for idx in sub_indices)

    def transfer_to_gpus(self, gpu_resources: List[Tuple[int, 'faiss.StandardGpuResources']]):
        """
        Transfer sub-indices to GPUs in round-robin fashion.

        Args:
            gpu_resources: List of (gpu_id, resources) tuples
        """
        import faiss

        self.gpu_indices = []
        num_gpus = len(gpu_resources)

        for i, sub_idx in enumerate(self.sub_indices):
            gpu_id, res = gpu_resources[i % num_gpus]

            try:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True  # 2x memory savings
                co.useFloat16LookupTables = True

                gpu_idx = faiss.index_cpu_to_gpu(res, gpu_id, sub_idx, co)
                self.gpu_indices.append((gpu_idx, gpu_id))
            except Exception as e:
                # Fall back to CPU for this chunk
                self.gpu_indices.append((sub_idx, -1))  # -1 = CPU

    def search(self, queries: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search all sub-indices and merge top-k results.

        Each sub-index is GPU-safe (<5K objects), and queries are batched
        to avoid CUBLAS crashes. Results are merged to find global top-k neighbors.
        """
        N = len(queries)

        # Batch queries to avoid CUBLAS crashes
        if N > self.MAX_QUERY_BATCH:
            # Process in batches
            all_D = []
            all_I = []
            for batch_start in range(0, N, self.MAX_QUERY_BATCH):
                batch_end = min(batch_start + self.MAX_QUERY_BATCH, N)
                batch_queries = queries[batch_start:batch_end]
                D_batch, I_batch = self._search_batch(batch_queries, k)
                all_D.append(D_batch)
                all_I.append(I_batch)
            return np.vstack(all_D), np.vstack(all_I)

        return self._search_batch(queries, k)

    def _search_batch(self, queries: np.ndarray, k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Search a single batch of queries (small enough for GPU)."""
        N = len(queries)

        # Use GPU indices if available, else CPU
        indices_to_search = self.gpu_indices if self.gpu_indices else [(idx, -1) for idx in self.sub_indices]

        all_distances = []
        all_indices = []

        for i, (idx, gpu_id) in enumerate(indices_to_search):
            offset = self.offsets[i]

            # Search this chunk
            D, I = idx.search(queries, k)

            # Convert local indices to global indices
            I_global = I.copy()
            valid_mask = I_global >= 0
            I_global[valid_mask] += offset

            all_distances.append(D)
            all_indices.append(I_global)

        # Merge results: for each query, get top-k across all chunks
        # Using vectorized numpy operations for speed
        merged_D = np.full((N, k), float('inf'), dtype=np.float32)
        merged_I = np.full((N, k), -1, dtype=np.int64)

        for D, I in zip(all_distances, all_indices):
            # Stack current merged with new results
            combined_D = np.hstack([merged_D, D])
            combined_I = np.hstack([merged_I, I])

            # Get top-k indices for each row
            top_k_cols = np.argpartition(combined_D, k-1, axis=1)[:, :k]

            # Sort the top-k by distance
            rows = np.arange(N)[:, None]
            top_k_D = combined_D[rows, top_k_cols]
            sort_within = np.argsort(top_k_D, axis=1)
            top_k_cols_sorted = top_k_cols[rows, sort_within]

            merged_D = combined_D[rows, top_k_cols_sorted]
            merged_I = combined_I[rows, top_k_cols_sorted]

        return merged_D, merged_I


def build_hierarchical_gpu_index(
    vectors: np.ndarray,
    N: int,
    dim: int,
    max_chunk_size: int = HierarchicalGPUIndex.MAX_GPU_SAFE_CHUNK,
    verbose: bool = True
) -> Tuple:
    """
    Build GPU-optimized hierarchical index.

    Splits large indices into GPU-safe chunks (5K objects each) to avoid
    CUBLAS batch matrix multiply crashes. Each chunk can be safely transferred
    to GPU and searched independently.

    Args:
        vectors: [N, dim] float32 vectors
        N: Number of vectors
        dim: Vector dimension
        max_chunk_size: Max objects per chunk (default from HierarchicalGPUIndex.MAX_GPU_SAFE_CHUNK)
        verbose: Print progress

    Returns:
        (index, is_hierarchical) tuple
    """
    import faiss

    if N <= max_chunk_size:
        # Small enough for single GPU-safe index
        # Use Flat for best accuracy on small sets
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_L2)
        index.train(vectors)
        index.add(vectors)
        return index, False

    # Build hierarchical index with GPU-safe chunks
    num_chunks = (N + max_chunk_size - 1) // max_chunk_size
    if verbose:
        print(f"    Building hierarchical GPU index: {N} objects → {num_chunks} chunks of {max_chunk_size}")

    sub_indices = []
    offsets = []

    for chunk_start in range(0, N, max_chunk_size):
        chunk_end = min(chunk_start + max_chunk_size, N)
        chunk_vectors = vectors[chunk_start:chunk_end]
        chunk_N = len(chunk_vectors)

        # Use Flat index for GPU efficiency (no IVF training overhead)
        index = faiss.index_factory(dim, "Flat", faiss.METRIC_L2)
        index.train(chunk_vectors)
        index.add(chunk_vectors)

        sub_indices.append(index)
        offsets.append(chunk_start)

        if verbose:
            chunk_num = len(sub_indices)
            print(f"      Chunk {chunk_num}/{num_chunks}: {chunk_N} objects → GPU-safe Flat")

    return HierarchicalGPUIndex(sub_indices, offsets), True


def _build_single_index_to_disk(args):
    """Worker function to build a single FAISS index and save to disk (for parallel execution)."""
    import faiss
    import math
    import time
    import os

    scale, vectors, N, dim, F, output_path = args

    # Smart caching: Check for fresh cache (< 24 hours old)
    cache_file = f"{os.path.dirname(output_path)}/index_scale_{scale}_N{N}.faiss"
    if os.path.exists(cache_file):
        cache_age = time.time() - os.path.getmtime(cache_file)
        if cache_age < 86400:  # Less than 24 hours old
            # Determine index type from existing file
            try:
                temp_index = faiss.read_index(cache_file)
                index_string = "Cached"
                del temp_index
                return scale, cache_file, F, index_string, N
            except Exception:
                pass  # Cache corrupted, rebuild

    # Create FAISS index
    # ALWAYS use Flat index to allow hierarchical GPU conversion at load time
    # Flat indices support get_xb() for vector extraction, IVF does not
    # Hierarchical GPU indexing will split large indices into 15K chunks for GPU safety
    index_string = "Flat"

    # Use 75% of vCPUs for FAISS (leave 25% for system overhead)
    # For 24 vCPUs: 18 threads gives 2-3x speedup on index building
    import os
    omp_threads = int(os.environ.get('OMP_NUM_THREADS', max(1, (os.cpu_count() or 8) * 3 // 4)))
    faiss.omp_set_num_threads(omp_threads)

    index = faiss.index_factory(dim, index_string, faiss.METRIC_L2)

    # Flat index: no training needed (just add vectors directly)
    # Training call is still required but is a no-op for Flat
    index.train(vectors)

    # Add vectors
    index.add(vectors)

    # Write to disk with caching (avoids pickle serialization)
    # Use cache_file path that includes N for cache key
    faiss.write_index(index, cache_file)

    return scale, cache_file, F, index_string, N


def build_faiss_indices_by_scale(
    objects: List['MusicalObject'],
    verbose: bool = True,
    parallel_workers: int = None,
    index_cache_dir: str = "/mnt/models/faiss_cache"
) -> Dict[int, Tuple['faiss.Index', Dict[int, 'MusicalObject'], int]]:
    """
    Build SEPARATE FAISS indices for each scale (parallel with file-based serialization).

    Why separate indices:
    1. Padding zeros distort L2 distance between different-length objects
    2. 64-step objects shouldn't match 2000-step objects (musically nonsensical)
    3. Reasonable dimensions allow PQ compression (8K-34K vs 266K)

    Parallelization strategy:
    - Workers build indices and write to disk (no pickle serialization)
    - Main process loads from disk (fast)
    - Gives 4x speedup for large corpora

    Args:
        objects: List of MusicalObject instances
        verbose: Print progress
        parallel_workers: Number of parallel workers (default: CPU count / 2)
        index_cache_dir: Directory to store temporary index files

    Returns:
        Dict[scale -> (index, index_to_object, F)]
          scale: Timestep length
          index: FAISS index for that scale
          index_to_object: Dict mapping index position to MusicalObject
          F: Feature dimension
    """
    import os
    import math
    import tempfile
    import shutil
    from concurrent.futures import ProcessPoolExecutor, as_completed

    try:
        import faiss
    except ImportError:
        raise ImportError("faiss required: pip install faiss-cpu or faiss-gpu")

    if parallel_workers is None:
        parallel_workers = max(1, (os.cpu_count() or 8) // 2)

    # Create cache directory
    os.makedirs(index_cache_dir, exist_ok=True)

    if verbose:
        print(f"\n{'='*70}")
        print("BUILDING FAISS INDICES BY SCALE (PARALLEL)")
        print(f"{'='*70}")
        print(f"  Parallel workers: {parallel_workers}")
        print(f"  Index cache: {index_cache_dir}")

    # Group objects by their timestep length (scale)
    by_scale = {}
    for obj in objects:
        L = obj.tensor.shape[0]
        if L not in by_scale:
            by_scale[L] = []
        by_scale[L].append(obj)

    if verbose:
        print(f"  Total objects: {len(objects)}")
        scales_list = sorted(by_scale.keys())
        print(f"  Unique scales: {len(scales_list)} ({scales_list[0]}-{scales_list[-1]} steps)")

    # Prepare tasks and metadata
    tasks = []
    scale_metadata = {}  # Store index_to_object mapping

    for scale in sorted(by_scale.keys()):
        scale_objects = by_scale[scale]
        N = len(scale_objects)

        if N < 2:
            continue

        F = scale_objects[0].tensor.shape[1]
        dim = scale * F

        # Flatten all objects at this scale
        vectors = np.zeros((N, dim), dtype=np.float32)
        index_to_object = {}

        for i, obj in enumerate(scale_objects):
            vectors[i] = obj.tensor.flatten()
            index_to_object[i] = obj

        output_path = os.path.join(index_cache_dir, f"index_scale_{scale}.faiss")
        tasks.append((scale, vectors, N, dim, F, output_path))
        scale_metadata[scale] = index_to_object

    num_scales = len(tasks)
    if verbose:
        print(f"\n  Building {num_scales} indices in parallel...", flush=True)

    # Build indices in parallel
    completed = 0
    index_paths = {}

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(_build_single_index_to_disk, task): task[0] for task in tasks}

        for future in as_completed(futures):
            try:
                result_scale, output_path, F, index_string, N = future.result()
                completed += 1
                index_paths[result_scale] = (output_path, F)

                if verbose and (completed <= 5 or completed % 10 == 0 or completed == num_scales):
                    print(f"    [{completed}/{num_scales}] Scale {result_scale}: {N} objects, {index_string}", flush=True)

            except Exception as e:
                scale = futures[future]
                print(f"    ERROR building index for scale {scale}: {e}", flush=True)

    # Load indices from disk
    if verbose:
        print(f"\n  Loading {len(index_paths)} indices from disk...", flush=True)

    indices = {}

    # Multi-GPU setup with proper memory management
    gpu_resources = []
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if verbose:
                print(f"  Found {num_gpus} GPUs available")

            for gpu_id in range(num_gpus):
                res = faiss.StandardGpuResources()
                res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp per GPU
                gpu_resources.append((gpu_id, res))

                if verbose:
                    print(f"    GPU {gpu_id}: Initialized with 2GB temp memory")
    except Exception as e:
        if verbose:
            print(f"  GPU: Not available ({e}), using CPU")

    # Load indices and use hierarchical GPU indexing for large scales
    GPU_SAFE_CHUNK_SIZE = HierarchicalGPUIndex.MAX_GPU_SAFE_CHUNK  # Use class constant (5000)
    scale_list = sorted(index_paths.keys())

    for i, scale in enumerate(scale_list):
        path, F = index_paths[scale]
        index_cpu = faiss.read_index(path)
        N = index_cpu.ntotal
        index_gpu = None

        # V11 FIX: Use SINGLE GPU index for ALL sizes, chunk queries not index
        # The CUBLAS error is from large query batches, not large indices
        # A 60K flat index on GPU works fine with query batch size <= 256
        if len(gpu_resources) > 0:
            gpu_id, res = gpu_resources[i % len(gpu_resources)]

            try:
                co = faiss.GpuClonerOptions()
                co.useFloat16 = True  # Use FP16 for 2x memory savings
                co.useFloat16LookupTables = True

                index_gpu = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu, co)

                if verbose:
                    print(f"    Scale {scale}: {N} objects → GPU {gpu_id}")
            except Exception as e:
                if verbose:
                    print(f"    Scale {scale}: {N} objects → CPU (GPU failed: {str(e)[:30]})")

        # Store: (gpu_index_or_None, cpu_index, metadata, F)
        indices[scale] = (index_gpu, index_cpu, scale_metadata[scale], F)

    if verbose:
        print(f"\n✓ Built {len(indices)} FAISS indices by scale (parallel)")

    return indices


# Keep old function for backwards compatibility but mark as deprecated
def build_faiss_index(
    objects: List['MusicalObject'],
    verbose: bool = True
) -> Tuple['faiss.Index', Dict[int, 'MusicalObject'], int, int]:
    """
    DEPRECATED: Use build_faiss_indices_by_scale instead.
    This function pads all objects to max length which distorts distances.
    """
    # Filter to only include objects at most common scale to avoid padding issues
    from collections import Counter
    scale_counts = Counter(obj.tensor.shape[0] for obj in objects)
    most_common_scale = scale_counts.most_common(1)[0][0]

    filtered_objects = [obj for obj in objects if obj.tensor.shape[0] == most_common_scale]

    if verbose:
        print(f"  WARNING: build_faiss_index is deprecated. Use build_faiss_indices_by_scale.")
        print(f"  Filtering to most common scale {most_common_scale}: {len(filtered_objects)}/{len(objects)} objects")

    indices = build_faiss_indices_by_scale(filtered_objects, verbose)
    if most_common_scale in indices:
        index, index_to_object, F = indices[most_common_scale]
        return index, index_to_object, most_common_scale, F
    else:
        raise ValueError("No valid index built")


def apply_inverse_transform_gpu(
    objects_gpu: torch.Tensor,
    transform: Dict,
    device: str = 'cuda'
) -> torch.Tensor:
    """
    Apply INVERSE of transform on GPU.

    For searching: Given target, find source such that transform(source) = target
    Equivalent to: source = inverse_transform(target)

    Args:
        objects_gpu: [N, L, F] tensor on GPU
        transform: {'name': str, 'amount': float}
        device: 'cuda' or 'cpu'

    Returns:
        inversed: [N, L, F] tensor on GPU
    """
    import torch.nn.functional as F

    name = transform['name']
    amount = transform['amount']
    result = objects_gpu.clone()

    if name == 'transpose_semitone':
        # Inverse: transpose by negative amount
        result[:, :, 0] = result[:, :, 0] - amount

    elif name == 'time_shift':
        # Inverse: shift by negative amount
        shifts = -int(amount)
        result = torch.roll(result, shifts=shifts, dims=1)

    elif name == 'time_scale':
        # Inverse: scale by reciprocal
        inv_scale = 1.0 / amount
        result = result.permute(0, 2, 1)
        result = F.interpolate(result, scale_factor=inv_scale, mode='nearest')
        result = result.permute(0, 2, 1)

    elif name == 'velocity_scale':
        # Inverse: scale by reciprocal
        result[:, :, 2] = result[:, :, 2] / amount

    elif name == 'inversion':
        # Self-inverse: inversion(inversion(x)) = x
        result[:, :, 0] = 2 * amount - result[:, :, 0]

    elif name == 'retrograde':
        # Self-inverse: reverse(reverse(x)) = x
        result = torch.flip(result, dims=[1])

    elif name.startswith('quantize'):
        # Quantization not easily reversible - skip
        pass

    elif name == 'concatenate' or name == 'concat_seq':
        # ConcatSeq (homogeneous case): Inverse takes first 1/n of target
        # CROSS-SCALE: Returns SHORTER tensor - caller must handle cross-scale search
        # This is the fast path for repetition patterns [x, x, x, ...]
        n_copies = int(amount)
        L = objects_gpu.shape[1]
        fragment_len = L // n_copies
        result = objects_gpu[:, :fragment_len, :].clone()

    else:
        # Unknown transform - return unchanged
        pass

    return result


def find_cross_piece_derivations_faiss(
    objects: List['MusicalObject'],
    faiss_index: 'faiss.Index',
    index_to_object: Dict[int, 'MusicalObject'],
    transforms: List[Dict],
    existing_graph: Dict['MusicalObject', 'Derivation'],
    max_error: float = 0.03,
    chunk_size: int = 50000,
    verbose: bool = True
) -> List[FAISSSearchResult]:
    """
    Find cross-piece derivations using FAISS approximate nearest neighbor.

    For each object (target):
      For each transform:
        1. Apply inverse transform to target
        2. Search FAISS for nearest neighbor (potential source)
        3. Compute actual error
        4. Track best (source, transform) pair

    Args:
        objects: All objects to test
        faiss_index: Trained FAISS index containing all objects as keys
        index_to_object: Mapping from FAISS index to MusicalObject
        transforms: List of transforms to test
        existing_graph: Current derivation graph
        max_error: Maximum MSE for valid derivation
        chunk_size: Process this many objects at once
        verbose: Print progress

    Returns:
        results: List of potential derivations (before MDL filtering)
    """
    import torch

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-PIECE DERIVATION SEARCH (FAISS)")
        print(f"{'='*70}")
        print(f"  Testing {len(objects)} objects with {len(transforms)} transforms")
        print(f"  Device: {device}")

    all_results = []
    total_processed = 0

    for chunk_start in range(0, len(objects), chunk_size):
        chunk_objects = objects[chunk_start:chunk_start + chunk_size]
        N = len(chunk_objects)

        # Prepare chunk for GPU
        max_len = max(obj.tensor.shape[0] for obj in chunk_objects)
        F = chunk_objects[0].tensor.shape[1]

        chunk_padded = np.zeros((N, max_len, F), dtype=np.float32)
        for i, obj in enumerate(chunk_objects):
            L = obj.tensor.shape[0]
            chunk_padded[i, :L, :] = obj.tensor

        chunk_gpu = torch.tensor(chunk_padded, dtype=torch.float32, device=device)

        # Track best derivation for each object in chunk
        best_errors = np.full(N, float('inf'), dtype=np.float32)
        best_sources = np.zeros(N, dtype=np.int32)
        best_transforms = np.zeros(N, dtype=np.int32)

        # Test each transform
        for t_idx, transform in enumerate(transforms):
            # Apply inverse transform on GPU
            inversed_gpu = apply_inverse_transform_gpu(chunk_gpu, transform, device)

            # Flatten for FAISS search: [N, L, F] -> [N, L*F]
            inversed_flat = inversed_gpu.reshape(N, -1).cpu().numpy().astype(np.float32)

            # FAISS search: Find nearest neighbor for each inversed target
            # This finds the best SOURCE for each target under this transform
            distances, indices = faiss_index.search(inversed_flat, k=1)

            # Convert L2 distance to MSE
            errors = distances[:, 0] / (max_len * F)

            # Update best if improved
            improved = errors < best_errors
            best_errors[improved] = errors[improved]
            best_sources[improved] = indices[improved, 0]
            best_transforms[improved] = t_idx

        # Collect valid results
        for i, target in enumerate(chunk_objects):
            if best_errors[i] < max_error:
                all_results.append(FAISSSearchResult(
                    target_idx=chunk_start + i,
                    source_idx=best_sources[i],
                    transform_idx=best_transforms[i],
                    error=best_errors[i],
                    distance=best_errors[i] * max_len * F
                ))

        total_processed += N
        if verbose and total_processed % 100000 == 0:
            print(f"  Processed {total_processed}/{len(objects)} objects...")

    if verbose:
        print(f"✓ Found {len(all_results)} potential cross-piece derivations")

    return all_results


def update_graph_with_shorter_paths(
    existing_graph: Dict['MusicalObject', 'Derivation'],
    search_results: List[FAISSSearchResult],
    objects: List['MusicalObject'],
    index_to_object: Dict[int, 'MusicalObject'],
    transforms: List[Dict],
    verbose: bool = True
) -> Tuple[Dict, Set, CrossPieceMDLStats]:
    """
    Update derivation graph with shorter paths from FAISS search.

    Compares path lengths and only replaces derivations when:
    1. Target was a source (new derivation)
    2. New path is shorter than existing (MDL improvement)

    Args:
        existing_graph: Current derivation graph
        search_results: Results from FAISS search
        objects: All objects (for target lookup)
        index_to_object: FAISS index to object mapping
        transforms: Transform library
        verbose: Print progress

    Returns:
        (updated_graph, sources, stats)
    """
    from discovery.emergent_hierarchy import Derivation

    if verbose:
        print(f"\n{'='*70}")
        print("MDL PATH COMPARISON & GRAPH UPDATE")
        print(f"{'='*70}")

    graph = dict(existing_graph)
    new_derivations = 0
    paths_shortened = 0

    for result in search_results:
        target = objects[result.target_idx]
        source = index_to_object[result.source_idx]
        transform = transforms[result.transform_idx]

        # Skip self-derivation
        if target == source:
            continue

        # Compute path length
        source_path_length = graph[source].path_length if source in graph else 0
        new_path_length = source_path_length + 1

        # Check if should add/replace
        should_add = False

        if target in graph:
            # Object already has derivation - compare path lengths
            current_path_length = graph[target].path_length
            if new_path_length < current_path_length:
                # SHORTER PATH FOUND!
                should_add = True
                paths_shortened += 1
        else:
            # Target was a source - now derived!
            should_add = True
            new_derivations += 1

        if should_add:
            graph[target] = Derivation(
                target=target,
                source=source,
                transform_name=transform['name'],
                transform_amount=transform['amount'],
                error=result.error,
                is_cross_track=(source.track_id != target.track_id),
                is_cross_section=(source.section_id != target.section_id if (source.section_id and target.section_id) else False),
                is_cross_piece=(source.piece_id != target.piece_id),
                path_length=new_path_length
            )

    # Recompute sources
    sources = set()
    for obj in objects:
        if obj not in graph:
            sources.add(obj)

    # Compute statistics
    if graph:
        avg_path_length = sum(d.path_length for d in graph.values()) / len(graph)
        total_description_length = sum(d.path_length for d in graph.values())
    else:
        avg_path_length = 0
        total_description_length = 0

    stats = CrossPieceMDLStats(
        iteration=0,  # Set by caller
        new_derivations=new_derivations,
        paths_shortened=paths_shortened,
        total_sources=len(sources),
        avg_path_length=avg_path_length,
        total_description_length=total_description_length,
        time_seconds=0  # Set by caller
    )

    if verbose:
        print(f"  New derivations (sources→derived): {new_derivations}")
        print(f"  Paths shortened (MDL improvement): {paths_shortened}")
        print(f"  Remaining sources: {len(sources)}")
        print(f"  Average path length: {avg_path_length:.2f}")

    return graph, sources, stats


def run_unified_faiss_discovery(
    objects: List['MusicalObject'],
    transforms: List[Dict],
    max_error: float = 0.03,
    max_iterations: int = 10,
    verbose: bool = True,
    enable_heterogeneous_concat: bool = True
) -> Tuple[Dict, Set, List[CrossPieceMDLStats]]:
    """
    Unified FAISS discovery - finds ALL derivations (within-piece AND cross-piece).

    REPLACES both iteration 1 AND cross-piece refinement.
    No artificial piece boundary in the search - more Lewinian.

    Two-phase discovery per iteration:
      Phase 1: Standard transforms + homogeneous ConcatSeq (FAST)
      Phase 2: Heterogeneous ConcatSeq for poorly-explained objects (SLOWER)

    Key insight: Skip CPU/GPU iteration 1 entirely. Use FAISS for everything.
    Memory efficient: Never computes N×N matrix, uses approximate NN instead.

    Args:
        objects: All musical objects (from all pieces/scales)
        transforms: Transform library (starts with 29 primitives)
        max_error: Maximum MSE for valid derivation
        max_iterations: Maximum discovery iterations
        verbose: Print progress
        enable_heterogeneous_concat: Run Phase 2 heterogeneous ConcatSeq discovery

    Returns:
        (final_graph, final_sources, iteration_stats)
    """
    import time
    from discovery.emergent_hierarchy import Derivation

    if verbose:
        print(f"\n{'='*70}", flush=True)
        print("UNIFIED FAISS DISCOVERY (SKIPPING ITERATION 1)", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Objects: {len(objects)}", flush=True)
        print(f"  Transforms: {len(transforms)}", flush=True)
        print(f"  Max iterations: {max_iterations}", flush=True)
        print(f"  Max error: {max_error}", flush=True)
        print(f"  Heterogeneous ConcatSeq: {enable_heterogeneous_concat}", flush=True)

    # Step 1: Build FAISS indices by scale (separate index per scale!)
    index_start = time.time()
    scale_indices = build_faiss_indices_by_scale(objects, verbose)
    index_time = time.time() - index_start

    if verbose:
        print(f"  Index build time: {index_time/60:.1f} minutes", flush=True)

    # Step 1.5: Generate valid splits for heterogeneous ConcatSeq (ONCE)
    valid_splits = None
    if enable_heterogeneous_concat:
        if verbose:
            print(f"\n{'='*70}", flush=True)
            print("GENERATING VALID SPLITS FOR HETEROGENEOUS CONCATSEQ", flush=True)
            print(f"{'='*70}", flush=True)

        available_scales = sorted(scale_indices.keys())
        valid_splits = generate_all_possible_splits(available_scales, max_components=4, early_stop=100)
        total_splits = sum(len(v) for v in valid_splits.values())

        if verbose:
            print(f"  Available scales: {available_scales[:10]}... ({len(available_scales)} total)", flush=True)
            print(f"  Generated {total_splits} valid split patterns", flush=True)
            print(f"  Target lengths with splits: {len(valid_splits)}", flush=True)

    # Step 2: Initialize empty graph (no within-piece first!)
    graph = {}
    sources = set(objects)  # Everything starts as source
    iteration_stats = []

    # Step 3: Iterative discovery
    for iteration in range(max_iterations):
        iter_start = time.time()

        if verbose:
            print(f"\n{'='*70}", flush=True)
            print(f"UNIFIED FAISS ITERATION {iteration + 1}/{max_iterations}", flush=True)
            print(f"{'='*70}", flush=True)
            print(f"  Current sources: {len(sources)}", flush=True)
            print(f"  Current derivations: {len(graph)}", flush=True)

        # =====================================================================
        # PHASE 1: Standard transforms + homogeneous ConcatSeq (FAST)
        # =====================================================================
        if verbose:
            print(f"\n  PHASE 1: Standard transforms + homogeneous concat...", flush=True)

        new_derivations, paths_shortened = find_all_derivations_faiss_by_scale(
            objects=objects,
            scale_indices=scale_indices,
            transforms=transforms,
            existing_graph=graph,
            max_error=max_error,
            verbose=verbose
        )

        # Update graph with Phase 1 results
        graph.update(new_derivations)
        phase1_new = len(new_derivations)

        if verbose:
            print(f"  Phase 1 results: {phase1_new} new derivations, {paths_shortened} paths shortened", flush=True)

        # =====================================================================
        # PHASE 2: Heterogeneous ConcatSeq for poorly-explained objects (SLOWER)
        # =====================================================================
        phase2_new = 0
        if enable_heterogeneous_concat and valid_splits:
            if verbose:
                print(f"\n  PHASE 2: Heterogeneous ConcatSeq discovery...", flush=True)

            # Get poorly explained objects (sources or high-error derivations)
            poorly_explained = get_poorly_explained_objects(objects, graph, error_threshold=max_error * 0.7)

            if verbose:
                print(f"  Poorly explained objects: {len(poorly_explained)}", flush=True)

            # Try heterogeneous ConcatSeq for each
            for i, obj in enumerate(poorly_explained):
                if verbose and i > 0 and i % 500 == 0:
                    print(f"    Progress: {i}/{len(poorly_explained)} ({phase2_new} found)", flush=True)

                result = discover_concat_seq(obj, scale_indices, valid_splits, max_error)

                if result is not None:
                    # Create derivation from ConcatSeqDerivation
                    # For now, use the first source as the "main" source
                    # and encode the pattern in transform_amount
                    primary_source = result.sources[0]
                    source_path_length = graph[primary_source].path_length if primary_source in graph else 0

                    # Check if this improves on existing
                    should_add = False
                    if obj not in graph:
                        should_add = True
                    elif source_path_length + 1 < graph[obj].path_length:
                        should_add = True
                        paths_shortened += 1

                    if should_add:
                        # Encode pattern: for homogeneous n=2 use amount=2, etc.
                        # For heterogeneous, encode as string of lengths
                        if result.is_homogeneous:
                            transform_amount = len(result.sources)
                        else:
                            # Encode as sum of lengths (simplified)
                            transform_amount = sum(result.component_lengths)

                        graph[obj] = Derivation(
                            target=obj,
                            source=primary_source,
                            transform_name='concat_seq',
                            transform_amount=transform_amount,
                            error=result.error,
                            is_cross_track=False,
                            is_cross_section=False,
                            is_cross_piece=(primary_source.piece_id != obj.piece_id),
                            path_length=source_path_length + 1
                        )
                        phase2_new += 1

            if verbose:
                print(f"  Phase 2 results: {phase2_new} new heterogeneous concat derivations", flush=True)

        # Update sources after both phases
        sources = {obj for obj in objects if obj not in graph}

        # Compute statistics
        total_new = phase1_new + phase2_new
        if graph:
            avg_path_length = sum(d.path_length for d in graph.values()) / len(graph)
            total_description_length = sum(d.path_length for d in graph.values())
        else:
            avg_path_length = 0
            total_description_length = 0

        iter_time = time.time() - iter_start

        stats = CrossPieceMDLStats(
            iteration=iteration + 1,
            new_derivations=total_new,
            paths_shortened=paths_shortened,
            total_sources=len(sources),
            avg_path_length=avg_path_length,
            total_description_length=total_description_length,
            time_seconds=iter_time
        )
        iteration_stats.append(stats)

        if verbose:
            print(f"\n  ITERATION {iteration + 1} SUMMARY:", flush=True)
            print(f"    Total new derivations: {total_new} (Phase1: {phase1_new}, Phase2: {phase2_new})", flush=True)
            print(f"    Paths shortened: {paths_shortened}", flush=True)
            print(f"    Sources remaining: {len(sources)}", flush=True)
            print(f"    Avg path length: {avg_path_length:.2f}", flush=True)
            print(f"    Iteration time: {iter_time/60:.1f} minutes", flush=True)

        # Check convergence
        if total_new == 0 and paths_shortened == 0:
            if verbose:
                print(f"\n{'='*70}", flush=True)
                print("CONVERGENCE REACHED", flush=True)
                print(f"{'='*70}", flush=True)
                print("No new derivations or path improvements found.", flush=True)
            break

    # Post-discovery analysis
    if verbose:
        total_time = index_time + sum(s.time_seconds for s in iteration_stats)
        print(f"\n{'='*70}", flush=True)
        print("UNIFIED FAISS DISCOVERY COMPLETE", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Final sources: {len(sources)}", flush=True)
        print(f"  Final derivations: {len(graph)}", flush=True)
        print(f"  Final avg path length: {iteration_stats[-1].avg_path_length:.2f}" if iteration_stats else "  No iterations", flush=True)
        print(f"  Total iterations: {len(iteration_stats)}", flush=True)
        print(f"  Total time: {total_time/60:.1f} minutes", flush=True)

        # Analyze discovered split patterns
        if enable_heterogeneous_concat:
            analyze_discovered_splits(graph, verbose=True)

    return graph, sources, iteration_stats


def find_all_derivations_faiss(
    objects: List['MusicalObject'],
    faiss_index: 'faiss.Index',
    index_to_object: Dict[int, 'MusicalObject'],
    max_len: int,
    F: int,
    transforms: List[Dict],
    existing_graph: Dict['MusicalObject', 'Derivation'],
    max_error: float = 0.03,
    chunk_size: int = 50000,
    verbose: bool = True
) -> Tuple[Dict['MusicalObject', 'Derivation'], int]:
    """
    Find ALL derivations using FAISS - no piece boundary restrictions.

    For each object (target):
      For each transform:
        1. Apply inverse transform to target
        2. Search FAISS for nearest neighbor (potential source)
        3. Check if this creates a shorter path (MDL)

    Args:
        objects: All objects to test
        faiss_index: Trained FAISS index
        index_to_object: Mapping from FAISS index to MusicalObject
        max_len: Maximum timesteps (from index building)
        F: Feature dimension (from index building)
        transforms: List of transforms to test
        existing_graph: Current derivation graph
        max_error: Maximum MSE for valid derivation
        chunk_size: Process this many objects at once
        verbose: Print progress

    Returns:
        (new_derivations, paths_shortened)
    """
    import torch
    from discovery.emergent_hierarchy import Derivation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"  Finding derivations with FAISS...")
        print(f"  Device: {device}")
        print(f"  Chunk size: {chunk_size}")
        print(f"  Index dimensions: {max_len} timesteps × {F} features")

    all_new_derivations = {}
    total_paths_shortened = 0
    total_processed = 0

    # Build object to index mapping for self-derivation check
    object_to_index = {obj: i for i, obj in enumerate(objects)}

    for chunk_start in range(0, len(objects), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(objects))
        chunk_objects = objects[chunk_start:chunk_end]
        N = len(chunk_objects)

        # Prepare chunk for GPU - pad to match index dimensions!
        chunk_padded = np.zeros((N, max_len, F), dtype=np.float32)
        for i, obj in enumerate(chunk_objects):
            L = obj.tensor.shape[0]
            chunk_padded[i, :L, :] = obj.tensor

        chunk_gpu = torch.tensor(chunk_padded, dtype=torch.float32, device=device)

        # Track best derivation for each object in chunk
        best_errors = np.full(N, float('inf'), dtype=np.float32)
        best_sources = np.zeros(N, dtype=np.int32)
        best_transforms = np.zeros(N, dtype=np.int32)

        # Test each transform
        for t_idx, transform in enumerate(transforms):
            # Apply inverse transform on GPU
            inversed_gpu = apply_inverse_transform_gpu(chunk_gpu, transform, device)

            # Flatten for FAISS search: [N, L, F] -> [N, L*F]
            inversed_flat = inversed_gpu.reshape(N, -1).cpu().numpy().astype(np.float32)

            # FAISS search: Find k=2 neighbors (in case first is self)
            distances, indices = faiss_index.search(inversed_flat, k=2)

            # For each object, pick best non-self neighbor
            for i in range(N):
                global_idx = chunk_start + i
                target = chunk_objects[i]

                # Check both neighbors
                for k in range(2):
                    source_idx = indices[i, k]
                    source = index_to_object[source_idx]

                    # Skip self-derivation
                    if source == target:
                        continue

                    # Convert L2 distance to MSE
                    error = distances[i, k] / (max_len * F)

                    if error < best_errors[i]:
                        best_errors[i] = error
                        best_sources[i] = source_idx
                        best_transforms[i] = t_idx
                    break  # Found valid non-self neighbor

        # Create derivations for valid matches
        for i, target in enumerate(chunk_objects):
            if best_errors[i] < max_error:
                source_idx = best_sources[i]
                source = index_to_object[source_idx]
                transform = transforms[best_transforms[i]]

                # Compute path length
                source_path_length = existing_graph[source].path_length if source in existing_graph else 0
                new_path_length = source_path_length + 1

                # Check if this improves on existing derivation
                should_add = False
                if target in existing_graph:
                    if new_path_length < existing_graph[target].path_length:
                        should_add = True
                        total_paths_shortened += 1
                else:
                    # Target was a source - now derived!
                    should_add = True

                if should_add:
                    all_new_derivations[target] = Derivation(
                        target=target,
                        source=source,
                        transform_name=transform['name'],
                        transform_amount=transform['amount'],
                        error=best_errors[i],
                        is_cross_track=(source.track_id != target.track_id),
                        is_cross_section=(source.section_id != target.section_id if (source.section_id and target.section_id) else False),
                        is_cross_piece=(source.piece_id != target.piece_id),
                        path_length=new_path_length
                    )

        total_processed += N
        if verbose and total_processed % 100000 == 0:
            print(f"    Processed {total_processed}/{len(objects)} objects...")

        # Free GPU memory
        del chunk_gpu, inversed_gpu
        torch.cuda.empty_cache()

    if verbose:
        print(f"  ✓ Found {len(all_new_derivations)} new derivations, {total_paths_shortened} paths shortened")

    return all_new_derivations, total_paths_shortened


def chunked_faiss_search(index, queries: np.ndarray, k: int = 2, chunk_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search FAISS index in chunks to avoid GPU CUBLAS errors.

    V12 FIX: The CUBLAS error (13) occurs with large (query × index × dim) products.
    With 60K index and 2128 dim, query batch must be <= 64 to avoid CUBLAS crash.
    Even 64 queries is fast on GPU - each batch takes ~0.1ms with GPU FAISS.

    Args:
        index: FAISS index (GPU or CPU) - can be any size
        queries: [N, D] query vectors
        k: Number of neighbors to return
        chunk_size: Maximum queries per batch (256 is safe for 60K index, 2048 dim)

    Returns:
        (distances, indices) - same as index.search() but chunked
    """
    N = len(queries)
    all_distances = []
    all_indices = []

    for i in range(0, N, chunk_size):
        chunk = queries[i:i+chunk_size]
        distances, indices = index.search(chunk, k)
        all_distances.append(distances)
        all_indices.append(indices)

    return np.vstack(all_distances), np.vstack(all_indices)


def find_all_derivations_faiss_by_scale(
    objects: List['MusicalObject'],
    scale_indices: Dict[int, Tuple['faiss.Index', Dict[int, 'MusicalObject'], int]],
    transforms: List[Dict],
    existing_graph: Dict['MusicalObject', 'Derivation'],
    max_error: float = 0.03,
    chunk_size: int = 10000,
    verbose: bool = True
) -> Tuple[Dict['MusicalObject', 'Derivation'], int]:
    """
    Find derivations using FAISS - searching within same scale only.

    This is the CORRECT approach:
    - 64-step objects only match other 64-step objects
    - No padding, no distance distortion
    - Musically meaningful (same temporal scale)

    Args:
        objects: All objects to test
        scale_indices: Dict[scale -> (index, index_to_object, F)]
        transforms: List of transforms to test
        existing_graph: Current derivation graph
        max_error: Maximum MSE for valid derivation
        chunk_size: Process this many objects at once
        verbose: Print progress

    Returns:
        (new_derivations, paths_shortened)
    """
    import torch
    from discovery.emergent_hierarchy import Derivation

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if verbose:
        print(f"  Finding derivations by scale with FAISS...")
        print(f"  Device: {device}")
        print(f"  Scales with indices: {sorted(scale_indices.keys())}")

    all_new_derivations = {}
    total_paths_shortened = 0

    # Group objects by scale
    objects_by_scale = {}
    for obj in objects:
        scale = obj.tensor.shape[0]
        if scale not in objects_by_scale:
            objects_by_scale[scale] = []
        objects_by_scale[scale].append(obj)

    # Track which scales have fallen back to CPU
    cpu_fallback_scales = set()

    # Process each scale independently
    for scale in sorted(scale_indices.keys()):
        if scale not in objects_by_scale:
            continue

        index_gpu, index_cpu, index_to_object, F = scale_indices[scale]
        scale_objects = objects_by_scale[scale]
        N = len(scale_objects)
        dim = scale * F

        # V11 FIX: Always use single GPU index (no hierarchical chunking)
        # Query batching (256 queries at a time) handles CUBLAS limits
        use_gpu = index_gpu is not None and scale not in cpu_fallback_scales
        index = index_gpu if use_gpu else index_cpu

        if verbose:
            mode = "GPU" if use_gpu else "CPU"
            idx_size = index_gpu.ntotal if index_gpu else index_cpu.ntotal
            print(f"\n  Processing scale {scale}: {N} objects, dim={dim}, index={idx_size} [{mode}]")

        # Process in chunks
        for chunk_start in range(0, N, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N)
            chunk_objects = scale_objects[chunk_start:chunk_end]
            chunk_N = len(chunk_objects)

            # Prepare chunk for GPU (no padding needed - all same scale!)
            chunk_padded = np.zeros((chunk_N, scale, F), dtype=np.float32)
            for i, obj in enumerate(chunk_objects):
                chunk_padded[i] = obj.tensor

            chunk_gpu = torch.tensor(chunk_padded, dtype=torch.float32, device=device)

            # Track best derivation for each object in chunk
            best_errors = np.full(chunk_N, float('inf'), dtype=np.float32)
            best_sources = np.zeros(chunk_N, dtype=np.int32)
            best_transforms = np.zeros(chunk_N, dtype=np.int32)

            # ============================================================
            # BATCHED TRANSFORM APPROACH: Apply ALL transforms on GPU first,
            # then ONE bulk transfer to CPU, then batch FAISS search
            # This reduces GPU<->CPU transfers from 33 to 1 per chunk!
            # ============================================================

            # Build list of valid transforms (skip time_scale)
            valid_transforms = [(t_idx, t) for t_idx, t in enumerate(transforms)
                               if t['name'] != 'time_scale']
            num_transforms = len(valid_transforms)

            # Apply all transforms on GPU and stack results
            # Shape: (num_transforms, chunk_N, scale, F)
            all_inversed_gpu = []
            transform_indices = []  # Track which transform each result corresponds to

            for t_idx, transform in valid_transforms:
                inversed_gpu = apply_inverse_transform_gpu(chunk_gpu, transform, device)
                expected_shape = (chunk_N, scale, F)
                if inversed_gpu.shape == expected_shape:
                    all_inversed_gpu.append(inversed_gpu)
                    transform_indices.append(t_idx)

            if not all_inversed_gpu:
                continue  # No valid transforms for this chunk

            # Stack all transformed tensors: (T, chunk_N, scale, F)
            stacked_gpu = torch.stack(all_inversed_gpu, dim=0)
            T = len(transform_indices)

            # ONE bulk GPU->CPU transfer instead of T transfers!
            # Reshape to (T * chunk_N, dim) for batched FAISS search
            stacked_flat = stacked_gpu.reshape(T * chunk_N, -1).cpu().numpy().astype(np.float32)

            # Free GPU memory immediately after transfer
            del stacked_gpu, all_inversed_gpu
            torch.cuda.empty_cache()

            # FAISS search: k neighbors for ALL transformed versions at once
            # V12 FIX: Use small query batch (64) to avoid CUBLAS errors
            # 64 queries × 60K index × 2128 dim is safe; 256 was crashing
            K_NEIGHBORS = 1
            search_chunk_size = 64  # Conservative to avoid CUBLAS errors with large indices

            try:
                # One big batched search: (T * chunk_N) queries
                distances_all, indices_all = chunked_faiss_search(
                    index, stacked_flat, k=K_NEIGHBORS, chunk_size=search_chunk_size
                )
            except Exception as e:
                if use_gpu and index_cpu is not None:
                    print(f"    [!] GPU search failed for scale {scale}, falling back to CPU: {str(e)[:50]}")
                    cpu_fallback_scales.add(scale)
                    index = index_cpu
                    use_gpu = False
                    distances_all, indices_all = chunked_faiss_search(
                        index, stacked_flat, k=K_NEIGHBORS, chunk_size=search_chunk_size
                    )
                else:
                    raise

            # Reshape results: (T, chunk_N, K)
            distances_all = distances_all.reshape(T, chunk_N, K_NEIGHBORS)
            indices_all = indices_all.reshape(T, chunk_N, K_NEIGHBORS)

            # Pre-build track_id lookup array (once per scale)
            CROSS_TRACK_PREFERENCE = 1.2
            if not hasattr(find_all_derivations_faiss_by_scale, '_track_id_cache'):
                find_all_derivations_faiss_by_scale._track_id_cache = {}

            cache_key = (scale, id(index_to_object))
            if cache_key not in find_all_derivations_faiss_by_scale._track_id_cache:
                max_idx = max(index_to_object.keys()) + 1
                track_ids_arr = np.full(max_idx, -1, dtype=np.int32)
                obj_to_idx = {}
                for idx, obj in index_to_object.items():
                    track_ids_arr[idx] = hash(obj.track_id) % (2**31)
                    obj_to_idx[id(obj)] = idx
                find_all_derivations_faiss_by_scale._track_id_cache[cache_key] = (track_ids_arr, obj_to_idx)

            track_ids_lookup, obj_to_idx = find_all_derivations_faiss_by_scale._track_id_cache[cache_key]

            # Build target info once for this chunk
            target_indices_np = np.array([obj_to_idx.get(id(t), -2) for t in chunk_objects], dtype=np.int64)
            target_track_ids_np = np.array([hash(obj.track_id) % (2**31) for obj in chunk_objects], dtype=np.int32)

            # Process all transforms together - k=1 so simplified logic
            INF = 1e10

            for t_local, t_idx in enumerate(transform_indices):
                # With k=1, distances/indices are shape (chunk_N, 1) - squeeze to (chunk_N,)
                distances = distances_all[t_local].squeeze(-1)  # (chunk_N,)
                indices_arr = indices_all[t_local].squeeze(-1)  # (chunk_N,)

                # Simple validation
                valid_mask = indices_arr >= 0
                errors = distances / dim
                error_valid = errors < max_error
                not_self = indices_arr != target_indices_np
                all_valid = valid_mask & error_valid & not_self

                # Check if better than current best
                improved = all_valid & (errors < best_errors)
                best_errors = np.where(improved, errors, best_errors)
                best_sources = np.where(improved, indices_arr, best_sources)
                best_transforms = np.where(improved, t_idx, best_transforms)

            # Create derivations for valid matches
            for i, target in enumerate(chunk_objects):
                if best_errors[i] < max_error:
                    source_idx = best_sources[i]

                    # Skip if no valid source found (FAISS returned -1)
                    if source_idx == -1:
                        continue

                    source = index_to_object[source_idx]
                    transform = transforms[best_transforms[i]]

                    # Compute path length
                    source_path_length = existing_graph[source].path_length if source in existing_graph else 0
                    new_path_length = source_path_length + 1

                    # Check if this improves on existing derivation
                    should_add = False
                    if target in existing_graph:
                        if new_path_length < existing_graph[target].path_length:
                            should_add = True
                            total_paths_shortened += 1
                    else:
                        # Target was a source - now derived!
                        should_add = True

                    if should_add:
                        all_new_derivations[target] = Derivation(
                            target=target,
                            source=source,
                            transform_name=transform['name'],
                            transform_amount=transform['amount'],
                            error=best_errors[i],
                            is_cross_track=(source.track_id != target.track_id),
                            is_cross_section=(source.section_id != target.section_id if (source.section_id and target.section_id) else False),
                            is_cross_piece=(source.piece_id != target.piece_id),
                            path_length=new_path_length
                        )

            # Free GPU memory (chunk_gpu only - transform results already freed after batch transfer)
            del chunk_gpu
            torch.cuda.empty_cache()

        if verbose:
            scale_derivations = sum(1 for d in all_new_derivations.values()
                                   if d.target.tensor.shape[0] == scale)
            print(f"    Scale {scale}: {scale_derivations} derivations found")

    if verbose:
        print(f"\n  ✓ Total: {len(all_new_derivations)} new derivations, {total_paths_shortened} paths shortened")

    return all_new_derivations, total_paths_shortened


def run_cross_piece_mdl(
    objects: List['MusicalObject'],
    transforms: List[Dict],
    existing_graph: Dict['MusicalObject', 'Derivation'],
    existing_sources: Set['MusicalObject'],
    max_error: float = 0.03,
    max_iterations: int = 10,
    verbose: bool = True
) -> Tuple[Dict, Set, List[CrossPieceMDLStats]]:
    """
    Run iterative cross-piece MDL optimization with FAISS.

    1. Build FAISS index (once)
    2. Iteratively find cross-piece derivations
    3. Update graph with shorter paths
    4. Converge when no improvements

    Args:
        objects: All musical objects
        transforms: Transform library
        existing_graph: Initial derivation graph (from within-piece)
        existing_sources: Initial sources
        max_error: Maximum MSE for valid derivation
        max_iterations: Maximum refinement iterations
        verbose: Print progress

    Returns:
        (final_graph, final_sources, iteration_stats)
    """
    import time

    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-PIECE MDL OPTIMIZATION (FAISS-ACCELERATED)")
        print(f"{'='*70}")
        print(f"  Objects: {len(objects)}")
        print(f"  Transforms: {len(transforms)}")
        print(f"  Initial sources: {len(existing_sources)}")
        print(f"  Max iterations: {max_iterations}")

    # Step 1: Build FAISS index (expensive, do once)
    index_start = time.time()
    faiss_index, index_to_object, max_len, F = build_faiss_index(objects, verbose)
    index_time = time.time() - index_start

    if verbose:
        print(f"  Index build time: {index_time/60:.1f} minutes")

    # Step 2: Iterative refinement
    graph = dict(existing_graph)
    sources = set(existing_sources)
    iteration_stats = []

    for iteration in range(max_iterations):
        iter_start = time.time()

        if verbose:
            print(f"\n{'='*70}")
            print(f"CROSS-PIECE ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")

        # Find derivations with FAISS
        search_results = find_cross_piece_derivations_faiss(
            objects=objects,
            faiss_index=faiss_index,
            index_to_object=index_to_object,
            transforms=transforms,
            existing_graph=graph,
            max_error=max_error,
            verbose=verbose
        )

        # Update graph with shorter paths
        graph, sources, stats = update_graph_with_shorter_paths(
            existing_graph=graph,
            search_results=search_results,
            objects=objects,
            index_to_object=index_to_object,
            transforms=transforms,
            verbose=verbose
        )

        iter_time = time.time() - iter_start
        stats.iteration = iteration + 1
        stats.time_seconds = iter_time
        iteration_stats.append(stats)

        if verbose:
            print(f"  Iteration time: {iter_time/60:.1f} minutes")

        # Check convergence
        if stats.new_derivations == 0 and stats.paths_shortened == 0:
            if verbose:
                print(f"\n{'='*70}")
                print("CONVERGENCE REACHED")
                print(f"{'='*70}")
                print("No new derivations or path improvements found.")
            break

    if verbose:
        print(f"\n{'='*70}")
        print("CROSS-PIECE MDL COMPLETE")
        print(f"{'='*70}")
        print(f"  Final sources: {len(sources)} (reduced from {len(existing_sources)})")
        print(f"  Final avg path length: {iteration_stats[-1].avg_path_length:.2f}")
        print(f"  Total iterations: {len(iteration_stats)}")

    return graph, sources, iteration_stats


# =============================================================================
# ConcatSeq: Two-Phase Discovery (Homogeneous + Heterogeneous)
# =============================================================================
#
# ConcatSeq is the universal concatenation primitive that handles both:
#   - Homogeneous: [x, x, x] - repetition (FAST to discover)
#   - Heterogeneous: [x, y, z] - composition (SLOWER, requires multi-query)
#
# Discovery Strategy:
#   Phase 1 (FAST): Try homogeneous patterns [x,x], [x,x,x], [x,x,x,x]
#   Phase 2 (SLOW): Try heterogeneous patterns if Phase 1 fails
# =============================================================================


@dataclass
class ConcatSeqDerivation:
    """Result from ConcatSeq discovery."""
    target: 'MusicalObject'
    sources: List['MusicalObject']  # List of source objects
    component_lengths: List[int]    # Length of each component
    error: float
    is_homogeneous: bool            # True if all sources are identical


def discover_concat_seq_homogeneous(
    target: 'MusicalObject',
    scale_indices: Dict[int, Tuple['faiss.Index', Dict[int, 'MusicalObject'], int]],
    max_error: float = 0.03,
    repetition_factors: List[int] = [2, 3, 4, 8]
) -> Optional[ConcatSeqDerivation]:
    """
    Phase 1: Discover homogeneous ConcatSeq patterns (FAST).

    Try patterns like [x, x], [x, x, x], [x, x, x, x].
    This is MUCH faster than heterogeneous because:
      - Only ONE FAISS query per repetition factor
      - Exact divisibility check as early filter

    Args:
        target: Target object to find derivation for
        scale_indices: Dict[scale -> (index, index_to_object, F)]
        max_error: Maximum MSE for valid derivation
        repetition_factors: List of n values to try

    Returns:
        ConcatSeqDerivation if found, None otherwise
    """
    target_len = target.tensor.shape[0]
    F = target.tensor.shape[1]

    for n in repetition_factors:
        # Early filter: target length must be divisible by n
        if target_len % n != 0:
            continue

        component_len = target_len // n

        # Check if we have an index for this component length
        if component_len not in scale_indices:
            continue

        index_gpu, index_cpu, index_to_object, _ = scale_indices[component_len]
        index = index_gpu if index_gpu is not None else index_cpu

        # Extract first component as query
        query = target.tensor[:component_len].flatten().astype(np.float32)
        query = query.reshape(1, -1)

        # Single FAISS query
        distances, indices = index.search(query, k=1)
        source_idx = indices[0, 0]
        source = index_to_object[source_idx]

        # Reconstruct by repeating
        reconstructed = np.tile(source.tensor, (n, 1))

        # Compute error
        error = np.mean((reconstructed - target.tensor) ** 2)

        if error < max_error:
            return ConcatSeqDerivation(
                target=target,
                sources=[source] * n,  # Same source repeated
                component_lengths=[component_len] * n,
                error=error,
                is_homogeneous=True
            )

    return None  # No homogeneous match found


def discover_concat_seq_heterogeneous(
    target: 'MusicalObject',
    scale_indices: Dict[int, Tuple['faiss.Index', Dict[int, 'MusicalObject'], int]],
    valid_splits: Dict[int, List[List[int]]],
    max_error: float = 0.03
) -> Optional[ConcatSeqDerivation]:
    """
    Phase 2: Discover heterogeneous ConcatSeq patterns (SLOWER).

    Try patterns like [x, y], [x, y, z] where sources are different.
    This requires MULTIPLE FAISS queries per split pattern.

    Args:
        target: Target object to find derivation for
        scale_indices: Dict[scale -> (index, index_to_object, F)]
        valid_splits: Dict[target_len -> List of valid split patterns]
        max_error: Maximum MSE for valid derivation

    Returns:
        ConcatSeqDerivation if found, None otherwise
    """
    target_len = target.tensor.shape[0]

    # Get valid split patterns for this target length
    if target_len not in valid_splits:
        return None

    for split_pattern in valid_splits[target_len]:
        # Check if we have indices for all component lengths
        all_scales_available = all(
            length in scale_indices for length in split_pattern
        )
        if not all_scales_available:
            continue

        # Split target into components and query each
        sources = []
        start = 0
        total_error = 0.0

        for length in split_pattern:
            # Extract component
            component = target.tensor[start:start + length]
            start += length

            # Query FAISS for this component
            index_gpu, index_cpu, index_to_object, _ = scale_indices[length]
            index = index_gpu if index_gpu is not None else index_cpu
            query = component.flatten().astype(np.float32).reshape(1, -1)

            distances, indices = index.search(query, k=1)
            source_idx = indices[0, 0]
            source = index_to_object[source_idx]
            sources.append(source)

            # Accumulate error
            component_error = distances[0, 0] / (length * target.tensor.shape[1])
            total_error += component_error * length

        # Compute average error
        avg_error = total_error / target_len

        if avg_error < max_error:
            # Check if actually heterogeneous (not all same source)
            unique_sources = set(id(s) for s in sources)
            is_homogeneous = len(unique_sources) == 1

            return ConcatSeqDerivation(
                target=target,
                sources=sources,
                component_lengths=split_pattern,
                error=avg_error,
                is_homogeneous=is_homogeneous
            )

    return None  # No heterogeneous match found


def discover_concat_seq(
    target: 'MusicalObject',
    scale_indices: Dict[int, Tuple['faiss.Index', Dict[int, 'MusicalObject'], int]],
    valid_splits: Optional[Dict[int, List[List[int]]]] = None,
    max_error: float = 0.03
) -> Optional[ConcatSeqDerivation]:
    """
    Two-phase ConcatSeq discovery: homogeneous first, then heterogeneous.

    Args:
        target: Target object to find derivation for
        scale_indices: Dict[scale -> (index, index_to_object, F)]
        valid_splits: Dict[target_len -> List of valid split patterns] (for heterogeneous)
        max_error: Maximum MSE for valid derivation

    Returns:
        ConcatSeqDerivation if found, None otherwise
    """
    # Phase 1: Try homogeneous patterns (FAST)
    result = discover_concat_seq_homogeneous(target, scale_indices, max_error)
    if result is not None:
        return result

    # Phase 2: Try heterogeneous patterns (SLOW)
    if valid_splits is not None:
        result = discover_concat_seq_heterogeneous(target, scale_indices, valid_splits, max_error)
        if result is not None:
            return result

    return None


def generate_dense_scales(
    min_scale: int = 16,
    max_scale: int = 512,
    grain: int = 2
) -> List[int]:
    """
    Generate dense uniform scales at perceptual grain size.

    Pure discovery approach: No musical assumptions about bar lengths.
    The grain size (2 = eighth note at 16th-note resolution) is the only
    assumption - it's the perceptual minimum, not a musical structure.

    Args:
        min_scale: Minimum scale (16 = 1 bar at 16th notes, ~1 second)
        max_scale: Maximum scale (512 = 32 bars)
        grain: Step size (2 = eighth note, finer than beat-level)

    Returns:
        List of scales: [16, 18, 20, 22, 24, 26, 28, 30, 32, ...]

    Why grain=2?
        - Finer than beat-level (4) to catch more boundary variations
        - Coarser than 16th-note (1) to keep object count manageable
        - 2 steps = 1 eighth note = good balance of density vs tractability
    """
    return list(range(min_scale, max_scale + 1, grain))


def generate_hybrid_scales(
    min_scale: int = 12,
    max_scale: int = 512,
    base_grain: int = 8
) -> List[int]:
    """
    Grain=8 base + strategic additions for common odd meters.

    A practical compromise between pure discovery (grain=2, 249 scales)
    and fast testing (grain=16, 32 scales).

    Captures:
    - All /4 meters: 3/4, 4/4, 5/4, 6/4, 7/4, 8/4, etc.
    - 6/8 (same as 3/4 at 12 steps)
    - Misses: 5/8, 7/8, 9/8, 11/8 (rare in big band repertoire)

    Args:
        min_scale: Minimum scale (default 12 for 3/4)
        max_scale: Maximum scale (512 = 32 bars)
        base_grain: Base step size (8 = half bar in 4/4)

    Returns:
        ~66 scales covering common Western meters

    Comparison:
        grain=2:  249 scales, 10+ hours
        grain=4:  125 scales, 5 hours
        hybrid:   ~66 scales, 2.75 hours
        grain=16:  32 scales, 1.5 hours (misses odd meters)
    """
    # Base: multiples of 8 starting from 16
    scales = set(range(16, max_scale + 1, base_grain))

    # Strategic additions for common odd meters
    critical_meters = [
        12,  # 3/4 AND 6/8 (both are 12 steps at 16th note resolution!)
        20,  # 5/4
        28,  # 7/4
    ]

    for meter in critical_meters:
        if meter >= min_scale and meter <= max_scale:
            scales.add(meter)

    return sorted(scales)


def generate_all_possible_splits(
    scales: List[int],
    max_components: int = 4,
    early_stop: int = 100
) -> Dict[int, List[List[int]]]:
    """
    Generate ALL possible splits from available scales.

    PURE DISCOVERY: No musical knowledge imposed.
    Structure emerges from which splits achieve low reconstruction error.

    Example discoveries:
        - If 44 = [28, 16] works often → "7/4 + 4/4 is real in this corpus"
        - If 44 = [22, 22] never works → "11/4 bars don't exist"
        - If 36 = [20, 16] works → "5/4 + 4/4 alternation exists"

    The meter structure is DISCOVERED, not imposed.

    Args:
        scales: Available extraction scales (e.g., from generate_dense_scales)
        max_components: Max components per split (2-4)
        early_stop: Max splits to keep per target length (prevents explosion)

    Returns:
        Dict[target_len -> List of split patterns]

    Computational cost for 128-step object with scales [16, 20, 24, ..., 124]:
        - ~28 2-splits to try
        - Each requires 2 FAISS queries
        - Total: 56 FAISS queries (fast!)
    """
    scale_set = set(scales)  # Fast lookup
    min_scale = min(scales)
    valid_splits = {}

    # 2-component splits (most common: A-B form)
    for target in scales:
        if target not in valid_splits:
            valid_splits[target] = []

        for a in scales:
            if a >= target:
                break
            b = target - a
            if b in scale_set and b >= min_scale:
                valid_splits[target].append([a, b])
                if len(valid_splits[target]) >= early_stop:
                    break

    # 3-component splits (A-B-C, A-B-A, etc.)
    if max_components >= 3:
        for target in scales:
            if len(valid_splits.get(target, [])) >= early_stop:
                continue

            if target not in valid_splits:
                valid_splits[target] = []

            for a in scales:
                if a >= target - 2 * min_scale:  # Need room for b, c
                    break
                for b in scales:
                    if a + b >= target:
                        break
                    c = target - a - b
                    if c in scale_set and c >= min_scale:
                        valid_splits[target].append([a, b, c])
                        if len(valid_splits[target]) >= early_stop:
                            break
                if len(valid_splits[target]) >= early_stop:
                    break

    # 4-component splits (A-B-A-B, A-B-C-D, etc.)
    if max_components >= 4:
        for target in scales:
            if len(valid_splits.get(target, [])) >= early_stop:
                continue

            if target not in valid_splits:
                valid_splits[target] = []

            for a in scales:
                if a >= target - 3 * min_scale:
                    break
                for b in scales:
                    if a + b >= target - 2 * min_scale:
                        break
                    for c in scales:
                        if a + b + c >= target:
                            break
                        d = target - a - b - c
                        if d in scale_set and d >= min_scale:
                            valid_splits[target].append([a, b, c, d])
                            if len(valid_splits[target]) >= early_stop:
                                break
                    if len(valid_splits[target]) >= early_stop:
                        break
                if len(valid_splits[target]) >= early_stop:
                    break

    return valid_splits


def get_poorly_explained_objects(
    objects: List['MusicalObject'],
    graph: Dict['MusicalObject', 'Derivation'],
    error_threshold: float = 0.02
) -> List['MusicalObject']:
    """
    Get objects that need heterogeneous ConcatSeq discovery.

    These are objects that either:
    1. Have no derivation (not in graph → sources)
    2. Have high reconstruction error (poorly explained by current derivation)

    Only these objects are tested with the expensive heterogeneous search.
    Most objects are already explained by homogeneous concat or other transforms.

    Args:
        objects: All musical objects
        graph: Current derivation graph {target: Derivation}
        error_threshold: Objects with error > this are considered poorly explained

    Returns:
        List of objects needing heterogeneous discovery
    """
    poorly_explained = []

    for obj in objects:
        if obj not in graph:
            # No derivation - this is a source, try to explain it
            poorly_explained.append(obj)
        elif graph[obj].error > error_threshold:
            # Has derivation but high error - try to find better
            poorly_explained.append(obj)

    return poorly_explained


def analyze_discovered_splits(
    graph: Dict['MusicalObject', 'Derivation'],
    verbose: bool = True
) -> Dict[Tuple[int, ...], int]:
    """
    Analyze which split patterns were discovered.

    This reveals the ACTUAL metric structure of the corpus,
    rather than imposing predefined bar units.

    Example output:
        {
          (32, 32): 892,     # Two 2-bar phrases
          (32, 16): 103,     # 2-bar phrase + 1-bar tag
          (28, 16): 47,      # 7/4 + 4/4 alternation (discovered!)
          (20, 16): 12,      # 5/4 + 4/4 alternation
        }

    Args:
        graph: Derivation graph after discovery
        verbose: Print analysis

    Returns:
        Dict[(component_lengths) -> frequency]
    """
    from collections import Counter

    split_counts = Counter()

    for target, derivation in graph.items():
        if derivation.transform_name == 'concat_seq':
            # Extract component lengths from transform params
            if hasattr(derivation, 'transform_params'):
                lengths = derivation.transform_params.get('lengths', [])
                if lengths:
                    split_counts[tuple(lengths)] += 1

    if verbose:
        print(f"\n{'='*70}")
        print("DISCOVERED SPLIT PATTERNS (Pure Discovery)")
        print(f"{'='*70}")
        print("These patterns were DISCOVERED, not imposed.\n")

        # Sort by frequency
        for pattern, count in split_counts.most_common(30):
            total = sum(pattern)
            components = ' + '.join(str(p) for p in pattern)
            print(f"  {total:4d} = [{components}]: {count} instances")

        # Infer bar units from high-frequency components
        component_counts = Counter()
        for pattern, count in split_counts.items():
            for component in pattern:
                component_counts[component] += count

        print(f"\n{'='*70}")
        print("INFERRED BAR UNITS (from frequency)")
        print(f"{'='*70}")
        for length, count in component_counts.most_common(15):
            # Interpret musically
            if length % 16 == 0:
                bars = length // 16
                meter = f"{bars}-bar 4/4"
            elif length % 12 == 0:
                bars = length // 12
                meter = f"{bars}-bar 3/4"
            elif length == 28:
                meter = "7/4 bar"
            elif length == 20:
                meter = "5/4 bar"
            else:
                beats = length / 4
                meter = f"{beats:.1f} beats"

            print(f"  {length:4d} timesteps ({meter}): {count} uses")

    return dict(split_counts)


# =============================================================================
# DEPRECATED: Old bar_units approach
# =============================================================================


def generate_valid_splits_predefined(
    max_length: int = 512,
    bar_units: List[int] = None,
    max_components: int = 4
) -> Dict[int, List[List[int]]]:
    """
    DEPRECATED: Generate splits from predefined bar units.

    This imposes musical structure a priori rather than discovering it.
    Use generate_all_possible_splits() with generate_dense_scales() instead.

    Kept for backwards compatibility and comparison studies.
    """
    if bar_units is None:
        bar_units = [12, 16, 20, 24, 28, 32, 48, 64, 96, 128]

    valid_splits = {}

    for a in bar_units:
        for b in bar_units:
            total = a + b
            if total <= max_length:
                if total not in valid_splits:
                    valid_splits[total] = []
                if [a, b] not in valid_splits[total]:
                    valid_splits[total].append([a, b])

    if max_components >= 3:
        for a in bar_units:
            for b in bar_units:
                for c in bar_units:
                    total = a + b + c
                    if total <= max_length:
                        if total not in valid_splits:
                            valid_splits[total] = []
                        if [a, b, c] not in valid_splits[total]:
                            valid_splits[total].append([a, b, c])

    if max_components >= 4:
        for a in bar_units:
            for b in bar_units:
                for c in bar_units:
                    for d in bar_units:
                        total = a + b + c + d
                        if total <= max_length:
                            if total not in valid_splits:
                                valid_splits[total] = []
                            if [a, b, c, d] not in valid_splits[total]:
                                valid_splits[total].append([a, b, c, d])

    return valid_splits


# Alias for backwards compatibility
def generate_valid_splits(
    scales: List[int] = None,
    max_components: int = 4,
    early_stop: int = 100
) -> Dict[int, List[List[int]]]:
    """
    Generate valid splits for heterogeneous ConcatSeq discovery.

    If scales is None, uses dense uniform scales (pure discovery).
    Otherwise uses the provided scales.

    This is the recommended entry point.
    """
    if scales is None:
        scales = generate_dense_scales()

    return generate_all_possible_splits(scales, max_components, early_stop)


# =============================================================================
# MDL Cost Function for ConcatSeq
# =============================================================================


def compute_concat_seq_mdl_cost(
    derivation: ConcatSeqDerivation,
    source_mdl_costs: Dict['MusicalObject', float],
    corpus_size: int = 10000
) -> float:
    """
    Compute MDL cost for a ConcatSeq derivation.

    Cost components:
      1. Source cost: Sum of MDL costs for unique sources
      2. Reference cost: Cost to point to each source (~log(corpus_size))
      3. Operation cost: Base cost for ConcatSeq operation
      4. Homogeneous bonus: Discount for repetition (simpler conceptually)

    Args:
        derivation: ConcatSeqDerivation to compute cost for
        source_mdl_costs: Dict mapping source objects to their MDL costs
        corpus_size: Size of corpus (for reference cost calculation)

    Returns:
        Total MDL cost in bits
    """
    import math

    # 1. Source cost: sum of costs for UNIQUE sources
    unique_sources = list(set(id(s) for s in derivation.sources))
    source_cost = sum(
        source_mdl_costs.get(s, 0.0)
        for s in derivation.sources
        if id(s) in unique_sources
    )

    # 2. Reference cost: pointing to each source in the corpus
    # ~log2(corpus_size) bits per reference
    reference_cost_per_source = math.log2(corpus_size) if corpus_size > 1 else 1.0
    reference_cost = len(derivation.sources) * reference_cost_per_source

    # 3. Operation cost: base cost for the ConcatSeq operation
    # Includes encoding the number of components and their lengths
    n_components = len(derivation.component_lengths)
    op_cost = 2.0 + math.log2(n_components + 1)  # Base + component count

    # 4. Homogeneous bonus: repetition is simpler than composition
    if derivation.is_homogeneous:
        # For homogeneous: only need to encode (source, n_repeats)
        # Much simpler than encoding n different sources
        homogeneous_bonus = -3.0 - math.log2(n_components)
    else:
        homogeneous_bonus = 0.0

    total_cost = source_cost + reference_cost + op_cost + homogeneous_bonus

    return total_cost
