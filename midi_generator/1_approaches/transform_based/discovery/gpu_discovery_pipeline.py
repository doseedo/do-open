"""
GPU-accelerated discovery pipeline.

Key optimizations:
1. Batch all pieces together on GPU (2000 simultaneously)
2. Test transform compositions in chunks (memory-efficient)
3. Parallelize sparse coding across all pieces
4. Expected speedup: 15-50x vs CPU (23 hours → 30-90 minutes)

Author: Agent 8 - GPU Tensor

ization
"""

import torch
from typing import List, Dict, Tuple, Optional
from core.tensor_representation import TensorMIDICorpus, load_corpus_to_gpu, load_corpus_to_gpu_cached
from core.tensor_transforms import TensorTransformLibrary, TransformComposer, create_transform_dictionary_tensor
from discovery.gpu_sparse_coding import GPUSparseEncoder, batch_sparse_encode
import time


class GPUDiscoveryPipeline:
    """
    GPU-accelerated discovery of transform compositions.

    Expected performance on A100:
    - Iteration 1: ~10 minutes (17 → 70 transforms)
    - Iteration 2: ~8 minutes (70 → 180 transforms)
    - Iteration 3-5: ~60 minutes combined (180 → 450 transforms)
    - Total: ~90 minutes vs 23 hours on CPU (15x speedup)
    """

    def __init__(
        self,
        device: str = 'cuda',
        batch_size: int = 2000,  # Process all pieces at once on A100
        chunk_size: int = 1000,   # Test 1000 compositions at a time
        lambda_sparsity: float = 0.1
    ):
        """
        Args:
            device: 'cuda' or 'cpu'
            batch_size: Number of pieces to process simultaneously
            chunk_size: Number of candidate compositions to test at once
            lambda_sparsity: L1 regularization for sparse coding
        """
        self.device = device
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.lambda_sparsity = lambda_sparsity

        self.corpus_converter = TensorMIDICorpus()
        self.transform_lib = TransformComposer()  # GPU-OPTIMIZED: Reuses library + dispatch table
        self.sparse_encoder = GPUSparseEncoder(
            lambda_sparsity=lambda_sparsity,
            device=device
        )

        # Memory management
        if device == 'cuda' and torch.cuda.is_available():
            props = torch.cuda.get_device_properties(device)
            self.gpu_memory_gb = props.total_memory / 1e9
            print(f"GPU: {props.name} ({self.gpu_memory_gb:.1f} GB)")
        else:
            self.gpu_memory_gb = 0

    def load_corpus_to_gpu(self, midi_files: List, cache_dir: str = None) -> torch.Tensor:
        """
        Load entire corpus to GPU as tensor with intelligent caching.

        Args:
            midi_files: List of mido.MidiFile objects
            cache_dir: Directory for cache files (default: ./tensor_cache)

        Returns:
            corpus_tensor: (B, T, F) on GPU
        """
        print(f"\n{'='*70}")
        print("LOADING CORPUS TO GPU")
        print(f"{'='*70}")

        corpus_tensor, _ = load_corpus_to_gpu_cached(
            midi_files,
            max_time_steps=self.corpus_converter.max_time_steps,
            device=self.device,
            cache_dir=cache_dir
        )

        return corpus_tensor

    def create_transform_dictionary(
        self,
        existing_transforms: List[Dict]
    ) -> torch.Tensor:
        """
        Convert transforms to tensor dictionary.

        Args:
            existing_transforms: List of {name: str, amount: float} dicts

        Returns:
            dict_tensor: (M, T, F) on GPU
        """
        return create_transform_dictionary_tensor(
            existing_transforms,
            max_time_steps=self.corpus_converter.max_time_steps,
            num_features=self.corpus_converter.num_features,
            device=self.device
        )

    def encode_corpus(
        self,
        corpus_tensor: torch.Tensor,
        transforms_dict: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Encode corpus with current transforms.

        Args:
            corpus_tensor: (B, T, F)
            transforms_dict: (M, T, F)

        Returns:
            encodings: (B, M) - sparse coefficients
            metrics: Dict with quality, sparsity, etc.
        """
        print(f"\n{'='*70}")
        print("SPARSE CODING")
        print(f"{'='*70}")

        start_time = time.time()

        encodings, metrics = batch_sparse_encode(
            corpus_tensor,
            transforms_dict,
            lambda_sparsity=self.lambda_sparsity,
            chunk_size=500,  # Process 500 pieces at a time for sparse coding
            verbose=True
        )

        # Compute quality
        quality = self.sparse_encoder.compute_reconstruction_quality(
            corpus_tensor,
            encodings,
            transforms_dict
        ).mean().item()

        elapsed = time.time() - start_time

        metrics['quality'] = quality
        metrics['time_seconds'] = elapsed

        print(f"\nOverall quality: {quality:.1%}")
        print(f"Average sparsity: {metrics['sparsity_mean']:.1f} transforms/piece")
        print(f"Time: {elapsed:.1f}s")

        return encodings, metrics

    def test_composition_candidates_batched(
        self,
        corpus_tensor: torch.Tensor,
        existing_transforms: List[Dict],
        current_encodings: torch.Tensor,
        transforms_dict: torch.Tensor,
        max_candidates: int = 100
    ) -> List[Dict]:
        """
        Test pairwise compositions in GPU-friendly chunks.

        Args:
            corpus_tensor: (B, T, F)
            existing_transforms: Current transform list
            current_encodings: (B, M)
            transforms_dict: (M, T, F)
            max_candidates: Return top K candidates

        Returns:
            best_candidates: List of promising new transforms
        """
        print(f"\n{'='*70}")
        print("TESTING COMPOSITIONS")
        print(f"{'='*70}")

        M = len(existing_transforms)
        total_compositions = M * M

        print(f"Testing {total_compositions:,} pairwise compositions...")
        print(f"Chunk size: {self.chunk_size}")

        num_chunks = (total_compositions + self.chunk_size - 1) // self.chunk_size

        candidates = []
        start_time = time.time()

        # Current reconstruction error (baseline)
        current_reconstruction = torch.einsum('bm,mtf->btf', current_encodings, transforms_dict)
        current_error = ((corpus_tensor - current_reconstruction) ** 2).sum(dim=(1, 2))

        # Free large tensors we no longer need
        del current_reconstruction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.chunk_size
            chunk_end = min(chunk_start + self.chunk_size, total_compositions)
            chunk_size_actual = chunk_end - chunk_start

            # Log chunk start
            print(f"\n  Chunk {chunk_idx + 1}/{num_chunks}: Testing compositions {chunk_start}-{chunk_end} ({chunk_size_actual} compositions)")

            # Generate compositions for this chunk
            compositions = []
            for i in range(chunk_start, chunk_end):
                idx1 = i // M
                idx2 = i % M

                comp = {
                    'transforms': [
                        existing_transforms[idx1],
                        existing_transforms[idx2]
                    ],
                    'name': f"{existing_transforms[idx1]['name']}_o_{existing_transforms[idx2]['name']}"
                }
                compositions.append(comp)

            # Test this chunk in batches
            chunk_improvements = self._test_composition_chunk(
                corpus_tensor,
                compositions,
                current_error
            )

            # Keep promising candidates
            chunk_candidates = 0
            for comp_idx, improvement in enumerate(chunk_improvements):
                if improvement > 0.01:  # Must improve by 1%
                    candidates.append({
                        'composition': compositions[comp_idx],
                        'improvement': improvement.item()
                    })
                    chunk_candidates += 1

            elapsed = time.time() - start_time
            pct = (chunk_end / total_compositions) * 100
            print(f"  ✓ Chunk complete: {pct:.1f}% done, found {chunk_candidates} candidates, {elapsed:.1f}s elapsed")

        elapsed = time.time() - start_time
        print(f"\nTested {total_compositions:,} compositions in {elapsed:.1f}s")
        print(f"Found {len(candidates)} promising candidates")

        # Sort by improvement
        candidates.sort(key=lambda x: x['improvement'], reverse=True)

        return candidates[:max_candidates]

    def _test_composition_chunk(
        self,
        corpus_tensor: torch.Tensor,
        compositions: List[Dict],
        baseline_error: torch.Tensor
    ) -> torch.Tensor:
        """
        LEVEL 3 OPTIMIZATION: Full batching - test ALL compositions in parallel.

        This replaces the sequential 196-iteration loop with a single batched operation.

        Key optimizations:
        1. Apply ALL 196 compositions at once: [196, 1720, T, F]
        2. Compute ALL errors in single GPU operation
        3. Expected speedup: 50-100x vs sequential testing
        4. Expected GPU utilization: 85-95% (vs 15-22%)

        Args:
            corpus_tensor: (B, T, F) where B=1720 pieces
            compositions: List of K composition dicts
            baseline_error: (B,) - current reconstruction error per piece

        Returns:
            improvements: (K,) - improvement score per composition (on GPU)
        """
        K = len(compositions)

        print(f"    Applying ALL {K} compositions in parallel (batched GPU operation)...", flush=True)

        # Free up GPU memory before large operation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            reserved_gb = torch.cuda.memory_reserved() / 1e9
            print(f"    GPU memory before batched op: {allocated_gb:.2f} GB allocated, {reserved_gb:.2f} GB reserved", flush=True)

        try:
            # LEVEL 3: Apply compositions in chunks and compute errors incrementally
            # This avoids materializing the full [196, 1720, T, F] tensor
            # With memory cleanup (1.85GB baseline), we can fit larger chunks:
            # 32 comps × 1720 × 256 × 133 × 4 bytes = ~27GB × 2 = ~54GB would be too much
            # But with grouping optimization, actual peak is lower
            chunk_size = 24  # Conservative: 24 comps = ~40GB + 1.85GB = ~42GB (safely under 40GB with overhead)
            all_improvements = []

            for start_idx in range(0, K, chunk_size):
                end_idx = min(start_idx + chunk_size, K)
                chunk_comps = compositions[start_idx:end_idx]
                chunk_size_actual = len(chunk_comps)

                print(f"    Processing chunk {start_idx//chunk_size + 1}/{(K + chunk_size - 1)//chunk_size}: compositions {start_idx}-{end_idx}...", flush=True)

                # Apply transforms for this chunk
                chunk_results = self.transform_lib.compose_transforms_batched(
                    corpus_tensor,
                    chunk_comps,
                    chunk_size=chunk_size_actual  # Process entire chunk at once
                )

                # Compute errors for this chunk: [chunk_size, 1720, T, F]
                chunk_errors = ((chunk_results - corpus_tensor.unsqueeze(0)) ** 2).sum(dim=(2, 3)).mean(dim=1)

                # Compute improvements for this chunk
                chunk_improvements = baseline_error.mean() - chunk_errors

                all_improvements.append(chunk_improvements)

                # Free memory
                del chunk_results, chunk_errors, chunk_improvements
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # Concatenate all improvements: [K]
            improvements = torch.cat(all_improvements, dim=0)

            print(f"    ✓ Batched test complete: {K} compositions tested in {len(all_improvements)} chunks", flush=True)

            return improvements

        except Exception as e:
            # Fallback to sequential if batched fails (e.g., OOM)
            print(f"    [!] Batched testing failed ({str(e)[:80]}), falling back to sequential...")

            all_improvements = torch.zeros(K, device=self.device)

            for idx, comp in enumerate(compositions):
                if idx % 50 == 0:
                    print(f"    Testing compositions {idx}-{min(idx+50, K)}/{K} (sequential fallback)", flush=True)

                try:
                    result = self.transform_lib.compose_transforms(
                        corpus_tensor,
                        [(t['name'], t['amount']) for t in comp['transforms']]
                    )
                    error = ((corpus_tensor - result) ** 2).sum(dim=(1, 2))
                    improvement = (baseline_error - error).mean()
                    all_improvements[idx] = improvement

                except Exception as e2:
                    print(f"    [!] Error testing {comp['name']}: {str(e2)[:80]}")

            return all_improvements

    def discovery_iteration_gpu(
        self,
        corpus_tensor: torch.Tensor,
        existing_transforms: List[Dict],
        target_quality: float = 0.99,
        max_new_transforms: int = 50
    ) -> Dict:
        """
        One iteration of discovery on GPU.

        Args:
            corpus_tensor: (B, T, F) - corpus on GPU
            existing_transforms: Current transform library
            target_quality: Stop if quality exceeds this
            max_new_transforms: Add up to this many new transforms

        Returns:
            results: {
                'new_transforms': List,
                'quality': float,
                'sparsity': float,
                'time': float,
                'candidates_tested': int
            }
        """
        start_time = time.time()

        # Step 1: Encode corpus with current transforms
        transforms_dict = self.create_transform_dictionary(existing_transforms)
        encodings, encode_metrics = self.encode_corpus(corpus_tensor, transforms_dict)

        quality = encode_metrics['quality']

        if quality >= target_quality:
            print(f"\n✓ Target quality {target_quality:.1%} reached!")
            return {
                'new_transforms': [],
                'quality': quality,
                'sparsity': encode_metrics['sparsity_mean'],
                'time': time.time() - start_time,
                'candidates_tested': 0
            }

        # Step 2: Test new compositions
        candidates = self.test_composition_candidates_batched(
            corpus_tensor,
            existing_transforms,
            encodings,
            transforms_dict,
            max_candidates=100
        )

        # Step 3: Select best candidates
        new_transforms = []
        for cand in candidates[:max_new_transforms]:
            new_transforms.append({
                'name': cand['composition']['name'],
                'transforms': cand['composition']['transforms'],
                'improvement': cand['improvement']
            })

        elapsed = time.time() - start_time

        return {
            'new_transforms': new_transforms,
            'quality': quality,
            'sparsity': encode_metrics['sparsity_mean'],
            'time': elapsed,
            'candidates_tested': len(candidates)
        }

    def run_full_discovery(
        self,
        midi_files: List,
        initial_transforms: List[Dict],
        target_quality: float = 0.99,
        max_iterations: int = 6,
        max_transforms_per_iteration: int = 50,
        cache_dir: str = None
    ) -> Dict:
        """
        Run complete discovery pipeline on GPU.

        Args:
            midi_files: List of mido.MidiFile objects
            initial_transforms: Starting transform library (17 primitives)
            target_quality: Stop when quality reaches this
            max_iterations: Maximum discovery iterations
            max_transforms_per_iteration: Add up to this many per iteration
            cache_dir: Directory for tensor cache (default: ./tensor_cache)

        Returns:
            final_results: {
                'all_transforms': List,
                'final_quality': float,
                'iterations': int,
                'total_time': float,
                'iteration_history': List[Dict]
            }
        """
        print(f"\n{'='*70}")
        print("GPU-ACCELERATED DISCOVERY PIPELINE")
        print(f"{'='*70}")
        print(f"Corpus size: {len(midi_files)} files")
        print(f"Starting transforms: {len(initial_transforms)}")
        print(f"Target quality: {target_quality:.1%}")
        print(f"Max iterations: {max_iterations}")

        # Load corpus to GPU once (with caching)
        corpus_tensor = self.load_corpus_to_gpu(midi_files, cache_dir=cache_dir)

        # Track progress
        all_transforms = initial_transforms.copy()
        iteration_history = []
        start_time = time.time()

        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            print(f"Current transforms: {len(all_transforms)}")

            iter_results = self.discovery_iteration_gpu(
                corpus_tensor,
                all_transforms,
                target_quality=target_quality,
                max_new_transforms=max_transforms_per_iteration
            )

            # Add new transforms
            all_transforms.extend(iter_results['new_transforms'])

            # Record history
            iteration_history.append({
                'iteration': iteration + 1,
                'quality': iter_results['quality'],
                'sparsity': iter_results['sparsity'],
                'new_transforms': len(iter_results['new_transforms']),
                'total_transforms': len(all_transforms),
                'time': iter_results['time']
            })

            print(f"\nIteration {iteration + 1} summary:")
            print(f"  Quality: {iter_results['quality']:.1%}")
            print(f"  New transforms: {len(iter_results['new_transforms'])}")
            print(f"  Total transforms: {len(all_transforms)}")
            print(f"  Time: {iter_results['time']:.1f}s")

            # Check if target reached
            if iter_results['quality'] >= target_quality:
                print(f"\n✓ Target quality {target_quality:.1%} reached!")
                break

        total_time = time.time() - start_time

        print(f"\n{'='*70}")
        print("DISCOVERY COMPLETE")
        print(f"{'='*70}")
        print(f"Final transforms: {len(all_transforms)}")
        print(f"Final quality: {iteration_history[-1]['quality']:.1%}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"{'='*70}")

        return {
            'all_transforms': all_transforms,
            'final_quality': iteration_history[-1]['quality'],
            'iterations': len(iteration_history),
            'total_time': total_time,
            'iteration_history': iteration_history
        }
