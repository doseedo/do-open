"""
Emergent hierarchy discovery via derivation graphs.

Key insight: Don't define hierarchy levels (segment/track/piece).
Instead, extract objects at multiple scales and let hierarchy emerge
from derivation relationships.

Algorithm:
    1. Extract objects at multiple scales (4-bar, 8-bar, full track, etc.)
    2. For each object, find best (source, transform) derivation
    3. Build derivation graph (nodes=objects, edges=transforms)
    4. Hierarchy emerges naturally:
       - Roots = irreducible sources (original musical content)
       - Depth = derivation chain length
       - Compositions = frequent paths in graph
       - Meta-patterns = repeated subgraphs

Example derivation graph:
    melody_bar_1 (SOURCE)
        ├─ Transpose(+7) → trumpet_bar_1
        │   └─ TimeShift(4) → trumpet_bar_5
        ├─ Harmonize(3rd) → sax_bar_1
        └─ Invert → bass_bar_1

Composition discovered: TimeShift(4) ∘ Transpose(+7)
(path from melody_bar_1 to trumpet_bar_5)

Author: Agent - Emergent Hierarchy
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict, Counter
import hashlib
from multiprocessing import Pool, cpu_count


# Global worker function for multiprocessing
def _process_piece_worker(args):
    """
    Worker function to process a single piece in parallel.
    Must be at module level for multiprocessing.Pool.
    """
    piece_objects, transforms, max_error = args
    from core.numpy_transforms import NumpyTransformLibrary

    lib = NumpyTransformLibrary()
    piece_graph = {}
    piece_sources = set()

    for target in piece_objects:
        # Get candidates within this piece
        candidates = [
            obj for obj in piece_objects
            if obj != target
            and (obj.end_time - obj.start_time) == (target.end_time - target.start_time)
            and (obj.track_id != target.track_id or obj.start_time < target.start_time)
        ]

        if not candidates:
            piece_sources.add(target)
            continue

        # Find best derivation
        best_source = None
        best_transform_name = None
        best_transform_amount = None
        best_error = float('inf')

        for source in candidates:
            for transform in transforms:
                try:
                    source_expanded = np.expand_dims(source.tensor, 0)
                    transformed = lib.apply_transform(
                        source_expanded,
                        transform['name'],
                        transform['amount']
                    )[0]

                    error = np.mean((target.tensor - transformed) ** 2)

                    if error < best_error:
                        best_error = error
                        best_source = source
                        best_transform_name = transform['name']
                        best_transform_amount = transform['amount']

                    if error < max_error * 0.01:
                        break
                except Exception:
                    continue

            if best_error < max_error * 0.01:
                break

        # Add to graph if valid
        if best_error < max_error and best_source is not None:
            from discovery.emergent_hierarchy import Derivation
            piece_graph[target] = Derivation(
                target=target,
                source=best_source,
                transform_name=best_transform_name,
                transform_amount=best_transform_amount,
                error=best_error,
                is_cross_track=(best_source.track_id != target.track_id),
                is_cross_section=(best_source.section_id != target.section_id if (best_source.section_id and target.section_id) else False),
                path_length=1  # Primitive transform = length 1
            )
        else:
            piece_sources.add(target)

    return piece_graph, piece_sources


def _process_piece_incremental_worker(args):
    """
    Worker function for incremental processing in parallel.
    Tests ALL objects with new transforms, replacing derivations when shorter path found.
    Must be at module level for multiprocessing.Pool.
    """
    objects_in_piece, piece_objects_all, transforms, max_error, existing_graph_piece = args
    from core.numpy_transforms import NumpyTransformLibrary

    lib = NumpyTransformLibrary()
    piece_new_derivations = {}
    piece_paths_shortened = 0  # Track MDL improvements

    for target in objects_in_piece:
        # Get candidates within this piece
        candidates = [
            obj for obj in piece_objects_all
            if obj != target
            and (obj.end_time - obj.start_time) == (target.end_time - target.start_time)
            and (obj.track_id != target.track_id or obj.start_time < target.start_time)
        ]

        if not candidates:
            continue  # Stays in sources

        # Find best derivation (only with new transforms)
        best_source = None
        best_transform_name = None
        best_transform_amount = None
        best_error = float('inf')

        for source in candidates:
            for transform in transforms:
                try:
                    source_expanded = np.expand_dims(source.tensor, 0)
                    transformed = lib.apply_transform(
                        source_expanded,
                        transform['name'],
                        transform['amount']
                    )[0]

                    error = np.mean((target.tensor - transformed) ** 2)

                    if error < best_error:
                        best_error = error
                        best_source = source
                        best_transform_name = transform['name']
                        best_transform_amount = transform['amount']

                    if error < max_error * 0.01:
                        break
                except Exception:
                    continue

            if best_error < max_error * 0.01:
                break

        # Add or replace derivation if valid
        if best_error < max_error and best_source is not None:
            from discovery.emergent_hierarchy import Derivation

            # Compute path_length based on source's path
            source_path_length = existing_graph_piece[best_source].path_length if best_source in existing_graph_piece else 0
            new_path_length = source_path_length + 1

            # Check if we should replace existing derivation (shorter path!)
            should_add = False
            if target in existing_graph_piece:
                # Object already has a derivation - compare path lengths
                current_path_length = existing_graph_piece[target].path_length
                if new_path_length < current_path_length:
                    # SHORTER PATH FOUND - this improves MDL!
                    should_add = True
                    piece_paths_shortened += 1
            else:
                # New derivation for a source object
                should_add = True

            if should_add:
                piece_new_derivations[target] = Derivation(
                    target=target,
                    source=best_source,
                    transform_name=best_transform_name,
                    transform_amount=best_transform_amount,
                    error=best_error,
                    is_cross_track=(best_source.track_id != target.track_id),
                    is_cross_section=(best_source.section_id != target.section_id if (best_source.section_id and target.section_id) else False),
                    path_length=new_path_length
                )

    return piece_new_derivations, piece_paths_shortened


@dataclass
class MusicalObject:
    """
    A musical object at any scale (segment, track, piece).

    The algorithm doesn't care what "level" it is - objects are just tensors
    with identity metadata.
    """
    piece_id: str
    track_id: str
    start_time: int
    end_time: int
    tensor: np.ndarray  # (T, F) musical content
    section_id: str = None  # For section-aware derivations (e.g., "verse_1", "chorus_1")
    is_drum: bool = False  # True if this is a drum/percussion track (MIDI channel 9)

    def __hash__(self):
        """Hash based on identity, not content"""
        return hash((self.piece_id, self.track_id, self.start_time, self.end_time))

    def __eq__(self, other):
        return (self.piece_id == other.piece_id and
                self.track_id == other.track_id and
                self.start_time == other.start_time and
                self.end_time == other.end_time)

    def __repr__(self):
        duration = self.end_time - self.start_time
        return f"Obj({self.piece_id}/{self.track_id}[{self.start_time}:{self.end_time}], {duration}steps)"


@dataclass
class Derivation:
    """A derivation: target = transform(source)"""
    target: MusicalObject
    source: MusicalObject
    transform_name: str
    transform_amount: float
    error: float
    is_cross_track: bool = False  # True if source.track_id != target.track_id
    is_cross_section: bool = False  # True if source.section_id != target.section_id
    is_cross_piece: bool = False  # True if source.piece_id != target.piece_id
    path_length: int = 1  # Number of primitive transforms in derivation path (for MDL)

    def __repr__(self):
        markers = []
        if self.is_cross_piece:
            markers.append("CROSS-PIECE")
        if self.is_cross_track:
            markers.append("CROSS-TRACK")
        if self.is_cross_section:
            markers.append("CROSS-SECTION")
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        return f"{self.target} = {self.transform_name}({self.transform_amount})({self.source}), err={self.error:.6f}, len={self.path_length}{marker_str}"


class EmergentHierarchyDiscovery:
    """
    Discovers musical structure via emergent hierarchy.

    No predefined levels - hierarchy emerges from derivation graph.
    """

    def __init__(
        self,
        scales: List[int] = [16, 32, 64, 128, 256],  # Segment sizes in timesteps
        max_error: float = 0.01,
        min_path_frequency: int = 3
    ):
        """
        Args:
            scales: Segment sizes to extract (timesteps at 16th note resolution)
                   [16=1bar, 32=2bars, 64=4bars, 128=8bars, 256=16bars, ...]
            max_error: Maximum error for valid derivation
            min_path_frequency: Minimum frequency for composition discovery
        """
        self.scales = scales
        self.max_error = max_error
        self.min_path_frequency = min_path_frequency

    def extract_objects(
        self,
        corpus: Dict,
        verbose: bool = True
    ) -> List[MusicalObject]:
        """
        Step 1: Extract objects at multiple scales.

        For each track:
          - Extract full track
          - Extract segments at various sizes (sliding window)

        Args:
            corpus: Hierarchical corpus {piece_id: {'tracks': {...}}}
            verbose: Print progress

        Returns:
            objects: List of MusicalObject at all scales
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STEP 1: MULTI-SCALE OBJECT EXTRACTION")
            print(f"{'='*70}")
            print(f"Scales: {self.scales} timesteps")

        objects = []

        for piece_id, piece_data in corpus.items():
            tracks = piece_data['tracks']
            track_metadata = piece_data.get('track_metadata', {})

            for track_id, track_tensor in tracks.items():
                T, F = track_tensor.shape

                # Get drum status from metadata
                track_meta = track_metadata.get(track_id, {})
                is_drum = track_meta.get('is_drum', False)

                # Full track
                objects.append(MusicalObject(
                    piece_id=piece_id,
                    track_id=track_id,
                    start_time=0,
                    end_time=T,
                    tensor=track_tensor,
                    is_drum=is_drum
                ))

                # Segments at each scale
                for scale in self.scales:
                    if scale >= T:
                        continue  # Skip if scale larger than track

                    # Sliding window
                    step = scale // 2  # 50% overlap
                    for start in range(0, T - scale + 1, step):
                        end = start + scale
                        segment = track_tensor[start:end]

                        # Assign section ID based on 256-timestep grid
                        # section_0: [0, 256), section_1: [256, 512), etc.
                        section_idx = start // 256
                        section_id = f"sect_{section_idx}"

                        objects.append(MusicalObject(
                            piece_id=piece_id,
                            track_id=track_id,
                            start_time=start,
                            end_time=end,
                            tensor=segment,
                            section_id=section_id,
                            is_drum=is_drum
                        ))

        if verbose:
            print(f"✓ Extracted {len(objects)} objects")
            scales_breakdown = Counter([obj.end_time - obj.start_time for obj in objects])
            print(f"  Breakdown by scale:")
            for scale in sorted(scales_breakdown.keys()):
                print(f"    {scale} steps: {scales_breakdown[scale]} objects")

        return objects

    def _get_constrained_candidates(
        self,
        target: MusicalObject,
        all_objects: List[MusicalObject],
        same_piece_only: bool = True
    ) -> List[MusicalObject]:
        """
        Get plausible source candidates for a target object.

        Constraints:
        1. Same piece (intra-piece derivation)
        2. Compatible size (same duration)
        3. Different track OR earlier segment in same track

        This reduces O(N²) to O(N × K) where K ~ 50-200 instead of N ~ 6254
        """
        target_size = target.end_time - target.start_time
        candidates = []

        for obj in all_objects:
            if obj == target:
                continue

            # Size compatibility
            if (obj.end_time - obj.start_time) != target_size:
                continue

            # Same piece constraint
            if same_piece_only and obj.piece_id != target.piece_id:
                continue

            # Different track OR earlier segment in same track
            if obj.track_id != target.track_id:
                candidates.append(obj)
            elif obj.start_time < target.start_time:  # Earlier segment
                candidates.append(obj)

        return candidates

    def build_derivation_graph(
        self,
        objects: List[MusicalObject],
        transforms: List[Dict],
        verbose: bool = True,
        use_gpu: bool = False,
        same_piece_only: bool = True,
        num_workers: Optional[int] = None,
        existing_graph: Optional[Dict[MusicalObject, Derivation]] = None,
        existing_sources: Optional[Set[MusicalObject]] = None,
        new_transforms_only: Optional[List[Dict]] = None
    ) -> Tuple[Dict[MusicalObject, Derivation], Set[MusicalObject]]:
        """
        Step 2: Build derivation graph (OPTIMIZED + PARALLEL + INCREMENTAL).

        For each object, find best (source, transform) derivation.
        Uses constraints to reduce from O(N²) to O(N × K).
        Parallelizes across pieces for massive speedup.

        INCREMENTAL MODE: If existing_graph is provided, only tests sources
        (not-yet-derived objects) with new_transforms_only.

        Args:
            objects: List of musical objects
            transforms: Available primitive transforms
            verbose: Print progress
            use_gpu: Use GPU acceleration if available (PyTorch)
            same_piece_only: Only derive within same piece (faster + enables parallelism)
            num_workers: Number of parallel workers (None = auto-detect CPU count)
            existing_graph: Previous derivation graph (for incremental mode)
            existing_sources: Previous source set (for incremental mode)
            new_transforms_only: Only test these transforms (for incremental mode)

        Returns:
            (graph, sources)
              graph: {target: Derivation(source, transform, error)}
              sources: Set of objects that can't be derived (roots)
        """
        # INCREMENTAL MODE: Test ALL objects with new transforms (for MDL path shortening)
        incremental_mode = (existing_graph is not None and
                          existing_sources is not None and
                          new_transforms_only is not None)

        if incremental_mode:
            # Start with existing graph and sources
            graph = dict(existing_graph)
            sources = set(existing_sources)
            objects_to_test = objects  # TEST ALL OBJECTS, not just sources (MDL improvement!)
            transforms_to_test = new_transforms_only  # Only test new transforms

            if verbose:
                print(f"\n{'='*70}")
                print("STEP 2: INCREMENTAL DERIVATION GRAPH UPDATE (MDL PATH SHORTENING)")
                print(f"{'='*70}")
                print(f"Testing ALL {len(objects_to_test)} objects for shorter paths")
                print(f"With {len(transforms_to_test)} new composition transforms")
                print(f"Existing derivations: {len(graph)}")
                print(f"Existing sources: {len(sources)}")
                print(f"Constraint: same_piece_only={same_piece_only}")
                if same_piece_only and num_workers is None:
                    print(f"Parallel workers: {cpu_count()} (auto-detected)")
                elif same_piece_only:
                    print(f"Parallel workers: {num_workers}")
        else:
            # Normal mode: test all objects with all transforms
            objects_to_test = objects
            transforms_to_test = transforms
            graph = {}
            sources = set()

            if verbose:
                print(f"\n{'='*70}")
                print("STEP 2: DERIVATION GRAPH CONSTRUCTION (OPTIMIZED + PARALLEL)")
                print(f"{'='*70}")
                print(f"Testing {len(objects)} objects...")
                print(f"Constraint: same_piece_only={same_piece_only}")
                print(f"GPU acceleration: {use_gpu}")
                if same_piece_only and num_workers is None:
                    print(f"Parallel workers: {cpu_count()} (auto-detected)")
                elif same_piece_only:
                    print(f"Parallel workers: {num_workers}")

        # Try GPU if requested
        if use_gpu:
            try:
                if incremental_mode:
                    # NEW: GPU incremental mode for MDL path shortening (1.15M objects!)
                    return self._build_derivation_graph_gpu_incremental(
                        objects_to_test, transforms_to_test, verbose,
                        same_piece_only, graph, sources
                    )
                else:
                    # Standard GPU mode for iteration 1
                    return self._build_derivation_graph_gpu(
                        objects, transforms, verbose, same_piece_only
                    )
            except ImportError:
                print("PyTorch not available, falling back to CPU")
                use_gpu = False

        # Parallel CPU implementation (when same_piece_only=True)
        if same_piece_only and (num_workers is None or num_workers > 1):
            if incremental_mode:
                graph, sources, paths_shortened = self._build_derivation_graph_parallel_incremental(
                    objects_to_test, transforms_to_test, verbose, num_workers,
                    graph, sources, objects  # Need all objects for candidate search
                )
                # Store paths_shortened for convergence check
                self._last_paths_shortened = paths_shortened
                return graph, sources
            else:
                return self._build_derivation_graph_parallel(
                    objects, transforms, verbose, num_workers
                )

        # Serial CPU implementation with constraints
        from core.numpy_transforms import NumpyTransformLibrary
        lib = NumpyTransformLibrary()

        # In incremental mode, graph and sources are already initialized
        if not incremental_mode:
            graph = {}
            sources = set()

        # Get all objects for candidate search (might be different from objects_to_test in incremental mode)
        all_objects = objects if not incremental_mode else objects

        for i, target in enumerate(objects_to_test):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(objects_to_test)} objects...")
                import sys
                sys.stdout.flush()

            # Get constrained candidates (KEY OPTIMIZATION)
            candidates = self._get_constrained_candidates(
                target, all_objects, same_piece_only
            )

            if not candidates:
                sources.add(target)
                continue

            # In incremental mode, start with existing best (if any)
            if incremental_mode and target in graph:
                # Already derived - skip it
                # This shouldn't happen since we only test sources, but check anyway
                continue

            # Find best derivation
            best_source = None
            best_transform_name = None
            best_transform_amount = None
            best_error = float('inf')

            for source in candidates:
                # Try all transforms
                for transform in transforms_to_test:
                    try:
                        # Apply transform
                        source_expanded = np.expand_dims(source.tensor, 0)
                        transformed = lib.apply_transform(
                            source_expanded,
                            transform['name'],
                            transform['amount']
                        )[0]

                        # Compute error
                        error = np.mean((target.tensor - transformed) ** 2)

                        if error < best_error:
                            best_error = error
                            best_source = source
                            best_transform_name = transform['name']
                            best_transform_amount = transform['amount']

                        # Early termination if very good match
                        if error < self.max_error * 0.01:
                            break

                    except Exception:
                        continue

                if best_error < self.max_error * 0.01:
                    break  # Found excellent match

            # Add to graph if valid derivation found
            if best_error < self.max_error and best_source is not None:
                graph[target] = Derivation(
                    target=target,
                    source=best_source,
                    transform_name=best_transform_name,
                    transform_amount=best_transform_amount,
                    error=best_error,
                    is_cross_track=(best_source.track_id != target.track_id),
                    is_cross_section=(best_source.section_id != target.section_id if (best_source.section_id and target.section_id) else False),
                    path_length=1  # Primitive transform = length 1
                )
                # Remove from sources if it was there
                sources.discard(target)
            else:
                sources.add(target)

        if verbose:
            print(f"\n✓ Built derivation graph")
            print(f"  Derivations: {len(graph)}")
            print(f"  Sources (roots): {len(sources)}")
            print(f"  Derivation rate: {len(graph)/len(objects)*100:.1f}%")
            import sys
            sys.stdout.flush()

        return graph, sources

    def _build_derivation_graph_gpu(
        self,
        objects: List[MusicalObject],
        transforms: List[Dict],
        verbose: bool,
        same_piece_only: bool
    ) -> Tuple[Dict[MusicalObject, Derivation], Set[MusicalObject]]:
        """
        GPU-accelerated derivation graph construction.

        Process each piece independently on GPU.
        """
        import torch
        from core.numpy_transforms import NumpyTransformLibrary

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lib = NumpyTransformLibrary()

        if verbose:
            print(f"  Using device: {device}")

        # Group objects by piece
        by_piece = {}
        for i, obj in enumerate(objects):
            if obj.piece_id not in by_piece:
                by_piece[obj.piece_id] = []
            by_piece[obj.piece_id].append((i, obj))

        graph = {}
        sources = set()
        total_processed = 0

        # Process each piece on GPU
        for piece_id, piece_objects in by_piece.items():
            indices = [idx for idx, obj in piece_objects]
            objs = [obj for idx, obj in piece_objects]
            N = len(objs)

            if N < 2:
                sources.add(objs[0])
                continue

            # Stack all objects: (N, max_len, F)
            max_len = max(obj.tensor.shape[0] for obj in objs)
            F = objs[0].tensor.shape[1]

            objects_padded = np.zeros((N, max_len, F))
            for i, obj in enumerate(objs):
                L = obj.tensor.shape[0]
                objects_padded[i, :L, :] = obj.tensor

            objects_gpu = torch.tensor(objects_padded, dtype=torch.float32, device=device)

            # Find best derivation for each object in this piece
            best_errors = torch.full((N,), float('inf'), device=device)
            best_sources = torch.zeros(N, dtype=torch.long, device=device)
            best_transforms = torch.zeros(N, dtype=torch.long, device=device)

            # Test each transform
            for t_idx, transform in enumerate(transforms):
                # Apply transform to all objects
                transformed_np = lib.apply_transform(
                    objects_padded,
                    transform['name'],
                    transform['amount']
                )
                transformed_gpu = torch.tensor(transformed_np, dtype=torch.float32, device=device)

                # Compute pairwise MSE: (N_target, N_source)
                targets = objects_gpu.unsqueeze(1)  # (N, 1, L, F)
                sources_t = transformed_gpu.unsqueeze(0)  # (1, N, L, F)

                errors = ((targets - sources_t) ** 2).mean(dim=(-2, -1))  # (N, N)

                # Mask self-derivation
                mask = torch.eye(N, device=device).bool()
                errors[mask] = float('inf')

                # Update best
                min_errors, min_sources = errors.min(dim=1)
                improved = min_errors < best_errors
                best_errors[improved] = min_errors[improved]
                best_sources[improved] = min_sources[improved]
                best_transforms[improved] = t_idx

            # Convert to graph
            for target_idx in range(N):
                if best_errors[target_idx] < self.max_error:
                    source_idx = best_sources[target_idx].item()
                    target_obj = objs[target_idx]
                    source_obj = objs[source_idx]
                    graph[target_obj] = Derivation(
                        target=target_obj,
                        source=source_obj,
                        transform_name=transforms[best_transforms[target_idx]]['name'],
                        transform_amount=transforms[best_transforms[target_idx]]['amount'],
                        error=best_errors[target_idx].item(),
                        is_cross_track=(source_obj.track_id != target_obj.track_id),
                        is_cross_section=(source_obj.section_id != target_obj.section_id if (source_obj.section_id and target_obj.section_id) else False),
                        path_length=1  # Primitive transform = length 1
                    )
                else:
                    sources.add(objs[target_idx])

            total_processed += N
            if verbose and total_processed % 500 == 0:
                print(f"  Processed {total_processed}/{len(objects)} objects...")
                import sys
                sys.stdout.flush()

        return graph, sources

    def _apply_transform_gpu(self, objects_gpu: 'torch.Tensor', transform: Dict) -> 'torch.Tensor':
        """
        Apply transform directly on GPU (no CPU roundtrip!).

        Args:
            objects_gpu: [N, L, F] tensor on GPU
            transform: {'name': str, 'amount': float}

        Returns:
            transformed: [N, L, F] tensor on GPU
        """
        import torch
        import torch.nn.functional as F

        name = transform['name']
        amount = transform['amount']
        result = objects_gpu.clone()

        if name == 'transpose_semitone':
            # Shift pitch column (index 0)
            result[:, :, 0] = result[:, :, 0] + amount

        elif name == 'time_shift':
            # Roll time axis
            shifts = int(amount)
            result = torch.roll(result, shifts=shifts, dims=1)

        elif name == 'time_scale':
            # Interpolate time dimension
            # [N, L, F] -> [N, F, L] for interpolate -> [N, F, L'] -> [N, L', F]
            result = result.permute(0, 2, 1)
            result = F.interpolate(result, scale_factor=amount, mode='nearest')
            result = result.permute(0, 2, 1)

        elif name == 'velocity_scale':
            # Scale velocity column (index 2)
            result[:, :, 2] = result[:, :, 2] * amount

        elif name == 'inversion':
            # Invert pitch around axis
            result[:, :, 0] = 2 * amount - result[:, :, 0]

        elif name == 'retrograde':
            # Reverse time
            result = torch.flip(result, dims=[1])

        elif name.startswith('quantize'):
            # Quantize timing (complex, skip for now or approximate)
            pass

        else:
            # Fallback to CPU for unknown transforms
            from core.numpy_transforms import NumpyTransformLibrary
            lib = NumpyTransformLibrary()
            result_np = lib.apply_transform(
                objects_gpu.cpu().numpy(),
                name,
                amount
            )
            result = torch.tensor(result_np, dtype=objects_gpu.dtype, device=objects_gpu.device)

        return result

    def _build_derivation_graph_gpu_incremental(
        self,
        objects_to_test: List[MusicalObject],
        new_transforms: List[Dict],
        verbose: bool,
        same_piece_only: bool,
        existing_graph: Dict[MusicalObject, Derivation],
        existing_sources: Set[MusicalObject]
    ) -> Tuple[Dict[MusicalObject, Derivation], Set[MusicalObject]]:
        """
        GPU-accelerated incremental derivation with MDL path shortening.

        Tests ALL objects (not just sources) with new composition transforms.
        Replaces derivations when shorter paths are found.

        Memory management:
        - Chunks of 50K objects = ~6.5GB per chunk
        - Error matrix 50K × 50K = ~10GB
        - Total ~17GB per chunk (fits A100 40GB)
        """
        import torch

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if verbose:
            print(f"  Using device: {device}")
            print(f"  GPU incremental mode: testing {len(objects_to_test)} objects")

        CHUNK_SIZE = 50000  # 50K objects per chunk

        # Start with existing graph
        graph = dict(existing_graph)
        sources = set(existing_sources)
        total_paths_shortened = 0

        # Group by piece if needed
        if same_piece_only:
            objects_by_piece = {}
            for obj in objects_to_test:
                if obj.piece_id not in objects_by_piece:
                    objects_by_piece[obj.piece_id] = []
                objects_by_piece[obj.piece_id].append(obj)
        else:
            objects_by_piece = {'all': objects_to_test}

        total_processed = 0

        for piece_id, piece_objects in objects_by_piece.items():
            # Process piece in chunks
            for chunk_start in range(0, len(piece_objects), CHUNK_SIZE):
                chunk = piece_objects[chunk_start:chunk_start + CHUNK_SIZE]
                N = len(chunk)

                if N < 1:
                    continue

                # Prepare chunk for GPU
                max_len = max(obj.tensor.shape[0] for obj in chunk)
                F = chunk[0].tensor.shape[1]

                chunk_padded = np.zeros((N, max_len, F))
                for i, obj in enumerate(chunk):
                    L = obj.tensor.shape[0]
                    chunk_padded[i, :L, :] = obj.tensor

                chunk_gpu = torch.tensor(chunk_padded, dtype=torch.float32, device=device)

                # Find best derivation for each object
                best_errors = torch.full((N,), float('inf'), device=device)
                best_sources = torch.zeros(N, dtype=torch.long, device=device)
                best_transforms = torch.zeros(N, dtype=torch.long, device=device)

                # Get candidates (within piece or all objects)
                if same_piece_only:
                    candidates = piece_objects
                else:
                    candidates = objects_to_test

                # Prepare candidates for GPU
                M = len(candidates)
                cand_padded = np.zeros((M, max_len, F))
                for i, obj in enumerate(candidates):
                    L = obj.tensor.shape[0]
                    cand_padded[i, :L, :] = obj.tensor

                cand_gpu = torch.tensor(cand_padded, dtype=torch.float32, device=device)

                # Test each transform
                for t_idx, transform in enumerate(new_transforms):
                    # Apply transform on GPU
                    transformed_gpu = self._apply_transform_gpu(cand_gpu, transform)

                    # Compute pairwise MSE: [N, M]
                    targets = chunk_gpu.unsqueeze(1)  # [N, 1, L, F]
                    sources_t = transformed_gpu.unsqueeze(0)  # [1, M, L, F]

                    errors = ((targets - sources_t) ** 2).mean(dim=(-2, -1))  # [N, M]

                    # Mask self-derivation if testing within same set
                    if same_piece_only and chunk is candidates:
                        for i in range(N):
                            global_idx = chunk_start + i
                            if global_idx < M:
                                errors[i, global_idx] = float('inf')

                    # Update best
                    min_errors, min_sources = errors.min(dim=1)
                    improved = min_errors < best_errors
                    best_errors[improved] = min_errors[improved]
                    best_sources[improved] = min_sources[improved]
                    best_transforms[improved] = t_idx

                # Convert to derivations and check for path improvements
                for i, target in enumerate(chunk):
                    if best_errors[i] < self.max_error:
                        source = candidates[best_sources[i].item()]
                        transform = new_transforms[best_transforms[i].item()]

                        # Compute path length
                        source_path_length = graph[source].path_length if source in graph else 0
                        new_path_length = source_path_length + 1

                        # Check if shorter path (MDL improvement!)
                        should_add = False
                        if target in graph:
                            current_path_length = graph[target].path_length
                            if new_path_length < current_path_length:
                                should_add = True
                                total_paths_shortened += 1
                        else:
                            should_add = True

                        if should_add:
                            graph[target] = Derivation(
                                target=target,
                                source=source,
                                transform_name=transform['name'],
                                transform_amount=transform['amount'],
                                error=best_errors[i].item(),
                                is_cross_track=(source.track_id != target.track_id),
                                is_cross_section=(source.section_id != target.section_id if (source.section_id and target.section_id) else False),
                                path_length=new_path_length
                            )
                            sources.discard(target)
                    else:
                        sources.add(target)

                total_processed += N
                if verbose and total_processed % 10000 == 0:
                    print(f"  Processed {total_processed}/{len(objects_to_test)} objects (GPU)...")
                    import sys
                    sys.stdout.flush()

        if verbose:
            print(f"\n✓ GPU incremental derivation complete")
            print(f"  Total derivations: {len(graph)}")
            print(f"  Sources: {len(sources)}")
            print(f"  Paths shortened (MDL): {total_paths_shortened}")
            import sys
            sys.stdout.flush()

        # Store for convergence check
        self._last_paths_shortened = total_paths_shortened

        return graph, sources

    def _build_derivation_graph_parallel(
        self,
        objects: List[MusicalObject],
        transforms: List[Dict],
        verbose: bool,
        num_workers: Optional[int] = None
    ) -> Tuple[Dict[MusicalObject, Derivation], Set[MusicalObject]]:
        """
        PARALLEL CPU derivation graph construction.

        Process each piece independently using multiprocessing.Pool.
        This is embarrassingly parallel - pieces don't depend on each other.
        """
        # Group objects by piece
        by_piece = {}
        for obj in objects:
            if obj.piece_id not in by_piece:
                by_piece[obj.piece_id] = []
            by_piece[obj.piece_id].append(obj)

        if verbose:
            print(f"  Processing {len(by_piece)} pieces in parallel...")
            import sys
            sys.stdout.flush()

        # Prepare arguments for each piece
        piece_args = [
            (piece_objects, transforms, self.max_error)
            for piece_objects in by_piece.values()
        ]

        # Process pieces in parallel
        if num_workers is None:
            num_workers = cpu_count()

        graph = {}
        sources = set()

        with Pool(num_workers) as pool:
            results = pool.map(_process_piece_worker, piece_args)

        # Merge results from all pieces
        for piece_graph, piece_sources in results:
            graph.update(piece_graph)
            sources.update(piece_sources)

        if verbose:
            print(f"\n✓ Built derivation graph (parallel)")
            print(f"  Derivations: {len(graph)}")
            print(f"  Sources (roots): {len(sources)}")
            print(f"  Derivation rate: {len(graph)/len(objects)*100:.1f}%")
            import sys
            sys.stdout.flush()

        return graph, sources

    def _build_derivation_graph_parallel_incremental(
        self,
        objects_to_test: List[MusicalObject],
        new_transforms: List[Dict],
        verbose: bool,
        num_workers: Optional[int],
        existing_graph: Dict[MusicalObject, Derivation],
        existing_sources: Set[MusicalObject],
        all_objects: List[MusicalObject]
    ) -> Tuple[Dict[MusicalObject, Derivation], Set[MusicalObject], int]:
        """
        PARALLEL CPU incremental derivation graph update with MDL path shortening.

        Tests ALL objects with new transforms, replacing when shorter paths found.
        """
        # Group objects to test by piece
        objects_by_piece_to_test = {}
        for obj in objects_to_test:
            if obj.piece_id not in objects_by_piece_to_test:
                objects_by_piece_to_test[obj.piece_id] = []
            objects_by_piece_to_test[obj.piece_id].append(obj)

        # Also group all objects by piece (for candidate search)
        objects_by_piece = {}
        for obj in all_objects:
            if obj.piece_id not in objects_by_piece:
                objects_by_piece[obj.piece_id] = []
            objects_by_piece[obj.piece_id].append(obj)

        # Group existing graph by piece
        existing_graph_by_piece = {}
        for obj, deriv in existing_graph.items():
            if obj.piece_id not in existing_graph_by_piece:
                existing_graph_by_piece[obj.piece_id] = {}
            existing_graph_by_piece[obj.piece_id][obj] = deriv

        if verbose:
            print(f"  Processing {len(objects_by_piece_to_test)} pieces in parallel...")
            import sys
            sys.stdout.flush()

        # Prepare arguments for each piece
        piece_args = [
            (objects_by_piece_to_test[piece_id],
             objects_by_piece[piece_id],
             new_transforms,
             self.max_error,
             existing_graph_by_piece.get(piece_id, {}))
            for piece_id in objects_by_piece_to_test.keys()
        ]

        # Process pieces in parallel
        if num_workers is None:
            num_workers = cpu_count()

        # Start with existing graph and sources
        graph = dict(existing_graph)
        sources = set(existing_sources)

        with Pool(num_workers) as pool:
            results = pool.map(_process_piece_incremental_worker, piece_args)

        # Merge results from all pieces
        total_paths_shortened = 0
        for piece_new_derivations, piece_paths_shortened in results:
            graph.update(piece_new_derivations)
            total_paths_shortened += piece_paths_shortened
            # Update sources: remove newly derived
            for obj in piece_new_derivations.keys():
                sources.discard(obj)

        if verbose:
            print(f"\n✓ Updated derivation graph (parallel incremental + MDL)")
            print(f"  Total derivations: {len(graph)}")
            print(f"  Sources (roots): {len(sources)}")
            print(f"  New derivations found: {len(graph) - len(existing_graph)}")
            print(f"  Paths shortened (MDL improvement): {total_paths_shortened}")
            print(f"  Derivation rate: {len(graph)/len(all_objects)*100:.1f}%")
            import sys
            sys.stdout.flush()

        return graph, sources, total_paths_shortened

    def find_compositions_from_paths(
        self,
        graph: Dict[MusicalObject, Derivation],
        verbose: bool = True
    ) -> Tuple[List[Tuple[str, int]], Dict]:
        """
        Step 3: Discover compositions from DIRECT SINGLE-EDGE relationships.

        CRITICAL FIX: Instead of extracting multi-hop paths (which create meaningless
        round-trips like "A→B→A"), we now extract direct arrangement relationships
        like "brass_1 = brass_0, octave down".

        This extracts:
          ✓ TrackDerive(brass_0→brass_1) ∘ T(-12)  "brass_1 is brass_0, octave down"
          ✓ SectionDerive(sect_0→sect_1) ∘ time_shift(-16)  "section relationship"

        NOT:
          ✗ TrackDerive(brass_1→brass_0) ∘ T(12) ∘ TrackDerive(brass_0→brass_1) ∘ T(-12)
            "meaningless round-trip"

        Args:
            graph: Derivation graph
            verbose: Print progress

        Returns:
            (compositions, temporal_data)
              compositions: [(composition_string, frequency), ...]
              temporal_data: empty dict (for compatibility)
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STEP 3: COMPOSITION DISCOVERY (Direct Single-Edge Extraction)")
            print(f"{'='*70}")

        from collections import Counter

        # Extract direct single-edge relationships
        direct_compositions = Counter()

        for target, derivation in graph.items():
            # Only extract cross-track/cross-section relationships (arranging patterns)
            if derivation.is_cross_track or derivation.is_cross_section:
                # Format the single-edge relationship
                if derivation.is_cross_track and derivation.is_cross_section:
                    # SectionTrackDerive: Cross-track AND cross-section
                    source_section = derivation.source.section_id or "unknown"
                    source_track = derivation.source.track_id
                    target_section = target.section_id or "unknown"
                    target_track = target.track_id
                    pattern = f"SectionTrackDerive({source_section}.{source_track}→{target_section}.{target_track}) ∘ {derivation.transform_name}({derivation.transform_amount})"
                elif derivation.is_cross_track:
                    # TrackDerive: Cross-track only (same section)
                    source_track = derivation.source.track_id
                    target_track = target.track_id
                    pattern = f"TrackDerive({source_track}→{target_track}) ∘ {derivation.transform_name}({derivation.transform_amount})"
                elif derivation.is_cross_section:
                    # SectionDerive: Same track, cross-section
                    source_section = derivation.source.section_id or "unknown"
                    target_section = target.section_id or "unknown"
                    pattern = f"SectionDerive({source_section}→{target_section}) ∘ {derivation.transform_name}({derivation.transform_amount})"

                direct_compositions[pattern] += 1

        if verbose:
            print(f"  Extracted {len(direct_compositions)} unique direct relationships")
            print(f"  Total instances: {sum(direct_compositions.values())}")

        # Convert to sorted list
        compositions = direct_compositions.most_common()

        # Empty temporal data (for compatibility with existing code)
        temporal_data = {
            'canonical_to_raw': {},
            'identity_processes': {}
        }

        if verbose:
            print(f"\nTop 20 direct arrangement patterns:")
            for i, (comp, freq) in enumerate(compositions[:20]):
                print(f"  {i+1}. {comp} (freq={freq})")

        return compositions, temporal_data

    def _get_path_to_root(
        self,
        obj: MusicalObject,
        graph: Dict[MusicalObject, Derivation],
        max_depth: int = 4  # Reduced from 10 to prevent degenerate long paths
    ) -> List[str]:
        """
        Trace path from object to root (source) with cycle detection.

        Returns path as list of transform strings.
        Example: ['Transpose(+7)', 'TimeShift(4)', 'Invert']

        IMPORTANT: Now includes cycle detection to prevent infinite loops
        from bidirectional edges like A → B → A → B → ...
        """
        path = []
        current = obj
        depth = 0
        visited_nodes = set()  # Track visited nodes to detect cycles

        while current in graph and depth < max_depth:
            # Cycle detection
            if current in visited_nodes:
                break  # Stop if we've seen this node before

            visited_nodes.add(current)

            derivation = graph[current]

            # Determine derivation type and format string accordingly
            if derivation.is_cross_track and derivation.is_cross_section:
                # SectionTrackDerive: Cross-track AND cross-section
                source_section = derivation.source.section_id or "unknown"
                source_track = derivation.source.track_id
                target_section = derivation.target.section_id or "unknown"
                target_track = derivation.target.track_id
                transform_str = f"SectionTrackDerive({source_section}.{source_track}→{target_section}.{target_track}) ∘ {derivation.transform_name}({derivation.transform_amount})"
            elif derivation.is_cross_track:
                # TrackDerive: Cross-track only (same section)
                source_track = derivation.source.track_id
                target_track = derivation.target.track_id
                transform_str = f"TrackDerive({source_track}→{target_track}) ∘ {derivation.transform_name}({derivation.transform_amount})"
            elif derivation.is_cross_section:
                # SectionDerive: Same track, cross-section
                source_section = derivation.source.section_id or "unknown"
                target_section = derivation.target.section_id or "unknown"
                transform_str = f"SectionDerive({source_section}→{target_section}) ∘ {derivation.transform_name}({derivation.transform_amount})"
            else:
                # Regular single-object transform
                transform_str = f"{derivation.transform_name}({derivation.transform_amount})"

            path.append(transform_str)
            current = derivation.source
            depth += 1

        return path

    def find_meta_patterns_from_subgraphs(
        self,
        graph: Dict[MusicalObject, Derivation],
        sources: Set[MusicalObject],
        verbose: bool = True
    ) -> List[Dict]:
        """
        Step 4: Discover meta-patterns from repeated subgraphs.

        A meta-pattern is a subgraph structure that appears multiple times
        with different parameter values.

        Example:
            Source → Transpose(+7) → A
            Source → Transpose(+3) → B
            Source → Transpose(0) → C

        Meta-pattern: Source → Transpose(X) → * where X ∈ {+7, +3, 0}

        Args:
            graph: Derivation graph
            sources: Set of source objects (roots)
            verbose: Print progress

        Returns:
            meta_patterns: [{'source': obj, 'transform_type': str, 'params': [...]}, ...]
        """
        if verbose:
            print(f"\n{'='*70}")
            print("STEP 4: META-PATTERN DISCOVERY (Subgraph Analysis)")
            print(f"{'='*70}")

        # Group derivations by source
        by_source = defaultdict(list)
        for target, derivation in graph.items():
            by_source[derivation.source].append(derivation)

        # Find patterns within each source's derivations
        meta_patterns = []

        for source, derivations in by_source.items():
            if len(derivations) < 3:  # Need multiple derivations
                continue

            # Group by transform type
            by_transform_type = defaultdict(list)
            for deriv in derivations:
                by_transform_type[deriv.transform_name].append(deriv)

            # Check each transform type for parameter variation
            for transform_type, type_derivations in by_transform_type.items():
                if len(type_derivations) >= 3:
                    params = [d.transform_amount for d in type_derivations]
                    targets = [d.target for d in type_derivations]

                    meta_pattern = {
                        'source': source,
                        'transform_type': transform_type,
                        'parameters': params,
                        'targets': targets,
                        'frequency': len(type_derivations)
                    }
                    meta_patterns.append(meta_pattern)

        if verbose:
            print(f"✓ Found {len(meta_patterns)} meta-patterns")
            print(f"\nTop 10 meta-patterns:")
            for i, pattern in enumerate(sorted(meta_patterns, key=lambda x: -x['frequency'])[:10]):
                print(f"  {i+1}. {pattern['source']} → {pattern['transform_type']}(X)")
                print(f"     X ∈ {pattern['parameters'][:5]}, freq={pattern['frequency']}")

        return meta_patterns

    def run_full_discovery(
        self,
        corpus: Dict,
        transforms: List[Dict],
        verbose: bool = True,
        use_gpu: bool = False,
        same_piece_only: bool = True,
        num_workers: Optional[int] = None,
        existing_objects: Optional[List[MusicalObject]] = None,
        existing_graph: Optional[Dict[MusicalObject, Derivation]] = None,
        existing_sources: Optional[Set[MusicalObject]] = None,
        new_transforms_only: Optional[List[Dict]] = None
    ) -> Dict:
        """
        Complete emergent hierarchy discovery with algebraic simplification.

        INCREMENTAL MODE: If existing_graph is provided, only tests sources
        with new_transforms_only (much faster for iteration 2+).

        Args:
            corpus: Hierarchical MIDI corpus
            transforms: Transform primitives (or full library)
            verbose: Print progress
            use_gpu: Use GPU acceleration if available
            same_piece_only: Only derive within same piece (faster)
            num_workers: Number of parallel workers
            existing_objects: Previous objects (for incremental mode)
            existing_graph: Previous graph (for incremental mode)
            existing_sources: Previous sources (for incremental mode)
            new_transforms_only: Only test these transforms (for incremental mode)

        Returns:
            results: {
                'objects': List[MusicalObject],
                'graph': Dict[target: Derivation],
                'sources': Set[MusicalObject],
                'compositions': List[(str, int)],
                'temporal_data': Dict (canonical_to_raw, identity_processes),
                'meta_patterns': List[Dict],
                'statistics': Dict
            }
        """
        # Step 1: Extract objects (or reuse existing)
        if existing_objects is not None:
            objects = existing_objects
            if verbose:
                print(f"\n{'='*70}")
                print("STEP 1: REUSING OBJECTS FROM PREVIOUS ITERATION")
                print(f"{'='*70}")
                print(f"✓ Reusing {len(objects)} objects")
        else:
            objects = self.extract_objects(corpus, verbose)

        # Step 2: Build graph (incremental if possible)
        graph, sources = self.build_derivation_graph(
            objects, transforms, verbose, use_gpu, same_piece_only, num_workers,
            existing_graph, existing_sources, new_transforms_only
        )

        # Step 3: Find compositions (now with canonicalization)
        compositions, temporal_data = self.find_compositions_from_paths(graph, verbose)

        # Step 4: Find meta-patterns
        meta_patterns = self.find_meta_patterns_from_subgraphs(graph, sources, verbose)

        # Compute statistics
        statistics = {
            'total_objects': len(objects),
            'total_derivations': len(graph),
            'total_sources': len(sources),
            'total_compositions': len(compositions),
            'total_meta_patterns': len(meta_patterns),
            'avg_graph_depth': self._compute_avg_depth(graph),
            'derivation_rate': len(graph) / len(objects) if objects else 0
        }

        if verbose:
            print(f"\n{'='*70}")
            print("DISCOVERY STATISTICS")
            print(f"{'='*70}")
            for key, value in statistics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")

        return {
            'objects': objects,
            'graph': graph,
            'sources': sources,
            'compositions': compositions,
            'temporal_data': temporal_data,  # NEW: stores both canonical and raw paths
            'meta_patterns': meta_patterns,
            'statistics': statistics
        }

    def _compute_avg_depth(self, graph: Dict) -> float:
        """Compute average depth of derivation chains"""
        depths = []
        for obj in graph.keys():
            depth = 0
            current = obj
            while current in graph and depth < 20:
                current = graph[current].source
                depth += 1
            depths.append(depth)
        return np.mean(depths) if depths else 0

    def discover_cross_track_relationships(
        self,
        objects: List[MusicalObject],
        transforms: List[Dict],
        verbose: bool = True,
        num_workers: Optional[int] = None
    ) -> List[Dict]:
        """
        Discover cross-track relationships (e.g., "trombone = trumpet - fifth").

        This is separate from the main derivation graph because it compares
        entire tracks or track segments across different instruments.

        Args:
            objects: All musical objects
            transforms: Available transforms
            verbose: Print progress
            num_workers: Parallel workers

        Returns:
            List of cross-track relationships with statistics
        """
        if verbose:
            print(f"\n{'='*70}")
            print("CROSS-TRACK RELATIONSHIP DISCOVERY")
            print(f"{'='*70}")

        from core.multitrack_primitives import find_cross_track_relationships, format_cross_track_relationship

        # Group objects by piece
        objects_by_piece = defaultdict(list)
        for obj in objects:
            objects_by_piece[obj.piece_id].append(obj)

        if verbose:
            print(f"Analyzing {len(objects_by_piece)} pieces for cross-track patterns...")

        # Discover relationships in each piece
        all_relationships = []

        if num_workers and num_workers > 1:
            # Parallel processing (TODO: implement parallel version)
            # For now, use serial
            pass

        # Serial processing
        for i, (piece_id, piece_objects) in enumerate(objects_by_piece.items()):
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{len(objects_by_piece)} pieces...")

            # Find relationships in this piece
            relationships = find_cross_track_relationships(
                piece_objects,
                transforms,
                max_error=self.max_error,
                same_section_only=False
            )

            all_relationships.extend(relationships)

        if verbose:
            print(f"\n✓ Found {len(all_relationships)} cross-track relationships")

        # Aggregate and count patterns
        from collections import Counter
        pattern_counts = Counter()

        for rel in all_relationships:
            pattern_key = (
                format_cross_track_relationship(rel),
                rel.transform_name,
                rel.transform_amount
            )
            pattern_counts[pattern_key] += 1

        # Sort by frequency
        top_patterns = pattern_counts.most_common(50)

        if verbose:
            print(f"\nTop 20 Cross-Track Patterns:")
            for i, ((pattern_desc, transform_name, amount), freq) in enumerate(top_patterns[:20], 1):
                print(f"  {i:2d}. {pattern_desc}")
                print(f"      Frequency: {freq} piece(s)")

        # Return structured results
        results = []
        for (pattern_desc, transform_name, amount), freq in top_patterns:
            results.append({
                'pattern': pattern_desc,
                'transform_name': transform_name,
                'transform_amount': amount,
                'frequency': freq
            })

        return results
