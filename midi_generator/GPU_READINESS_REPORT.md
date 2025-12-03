# GPU Readiness Report for MDL Path Shortening

## Current Status: ⚠️ PARTIALLY READY

### ✅ What Works (Iteration 1)

The existing GPU implementation (`_build_derivation_graph_gpu`) supports:
- **Basic derivation**: Tests primitives on GPU for iteration 1
- **Within-piece batching**: Processes each piece independently
- **Memory efficiency**: Pads objects to max length per piece
- **Pairwise MSE**: Computes all-to-all errors efficiently using broadcasting

**Code location**: `emergent_hierarchy.py:604-713`

**Current algorithm**:
```python
objects_gpu = torch.tensor(objects_padded, device='cuda')  # [N, L, F]

for transform in transforms:
    transformed = apply_transform(objects, transform)  # CPU! ⚠️
    transformed_gpu = torch.tensor(transformed, device='cuda')

    targets = objects_gpu.unsqueeze(1)  # [N, 1, L, F]
    sources = transformed_gpu.unsqueeze(0)  # [1, N, L, F]
    errors = ((targets - sources) ** 2).mean(dim=(-2, -1))  # [N, N]

    # Update best derivations
    ...
```

### ❌ What's Missing (Iteration 2+ with MDL)

**Problem**: GPU is **disabled for incremental mode**

```python
# Line 483 in emergent_hierarchy.py
if use_gpu and not incremental_mode:  # ⚠️ GPU SKIPPED FOR ITERATION 2+!
    return self._build_derivation_graph_gpu(...)
```

**Why disabled**: Comment says "GPU not optimized for small batches"
- But this is WRONG now with MDL path shortening!
- Iteration 2+ tests ALL 1.15M objects (not just 590 sources)
- This is a HUGE batch - perfect for GPU!

### Required Implementation

#### Option A: Quick Fix - Enable GPU for Incremental with Path Shortening

**File**: `emergent_hierarchy.py:483`

```python
# OLD (disabled)
if use_gpu and not incremental_mode:
    return self._build_derivation_graph_gpu(...)

# NEW (enable for MDL)
if use_gpu:
    if incremental_mode:
        return self._build_derivation_graph_gpu_incremental(
            objects_to_test, transforms_to_test, verbose,
            same_piece_only, existing_graph
        )
    else:
        return self._build_derivation_graph_gpu(
            objects, transforms, verbose, same_piece_only
        )
```

#### Option B: Full GPU Pipeline (Recommended)

Implement `_build_derivation_graph_gpu_incremental()` with proper memory management:

```python
def _build_derivation_graph_gpu_incremental(
    self,
    objects: List[MusicalObject],
    new_transforms: List[Dict],
    verbose: bool,
    same_piece_only: bool,
    existing_graph: Dict[MusicalObject, Derivation]
) -> Tuple[Dict, Set, int]:
    """
    GPU-accelerated incremental derivation with MDL path shortening.

    Memory management for A100 (40GB):
    - Objects: 1.15M × 256 × 128 × 4 bytes = 150GB (TOO BIG!)
    - Solution: Process in chunks of 50K objects = 6.5GB per chunk
    - Transforms: 100 compositions (negligible)
    - Error matrix: 50K × 50K × 4 bytes = 10GB per chunk
    """
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Key parameters
    CHUNK_SIZE = 50_000  # 6.5GB per chunk on GPU

    # Initialize outputs
    graph = dict(existing_graph)
    sources = set()
    total_paths_shortened = 0

    # Group by piece if same_piece_only
    if same_piece_only:
        objects_by_piece = group_by_piece(objects)
    else:
        objects_by_piece = {'all': objects}

    for piece_id, piece_objects in objects_by_piece.items():
        # Process piece in chunks
        for chunk_start in range(0, len(piece_objects), CHUNK_SIZE):
            chunk = piece_objects[chunk_start:chunk_start + CHUNK_SIZE]

            # Load chunk to GPU
            max_len = max(obj.tensor.shape[0] for obj in chunk)
            F = chunk[0].tensor.shape[1]

            chunk_padded = np.zeros((len(chunk), max_len, F))
            for i, obj in enumerate(chunk):
                L = obj.tensor.shape[0]
                chunk_padded[i, :L, :] = obj.tensor

            chunk_gpu = torch.tensor(chunk_padded, dtype=torch.float32, device=device)

            # Test each new composition
            best_errors = torch.full((len(chunk),), float('inf'), device=device)
            best_sources = torch.zeros(len(chunk), dtype=torch.long, device=device)
            best_transforms = torch.zeros(len(chunk), dtype=torch.long, device=device)

            for t_idx, transform in enumerate(new_transforms):
                # CRITICAL: Apply transforms on GPU, not CPU!
                transformed_gpu = apply_transform_gpu(chunk_gpu, transform)

                # Get candidate sources (within piece or global)
                if same_piece_only:
                    candidates = piece_objects
                else:
                    candidates = objects  # All objects globally

                # For memory, process candidates in sub-chunks
                for cand_start in range(0, len(candidates), CHUNK_SIZE):
                    cand_chunk = candidates[cand_start:cand_start + CHUNK_SIZE]

                    # Load candidates to GPU
                    cand_padded = pad_objects(cand_chunk, max_len, F)
                    cand_gpu = torch.tensor(cand_padded, dtype=torch.float32, device=device)

                    # Pairwise MSE: [N_targets, N_candidates]
                    targets = chunk_gpu.unsqueeze(1)  # [N, 1, L, F]
                    sources_t = transformed_gpu.unsqueeze(0)  # [1, M, L, F]

                    # Only compute against candidates, not all-to-all
                    errors = ((targets - cand_gpu.unsqueeze(0)) ** 2).mean(dim=(-2, -1))

                    # Update best
                    min_errors, min_sources = errors.min(dim=1)
                    improved = min_errors < best_errors
                    best_errors[improved] = min_errors[improved]
                    best_sources[improved] = (cand_start + min_sources[improved])
                    best_transforms[improved] = t_idx

            # Convert results and check for path improvements
            for i, target in enumerate(chunk):
                if best_errors[i] < self.max_error:
                    source = candidates[best_sources[i].item()]
                    transform = new_transforms[best_transforms[i].item()]

                    # Compute path length
                    source_path_length = existing_graph[source].path_length if source in existing_graph else 0
                    new_path_length = source_path_length + 1

                    # Check if shorter path
                    should_add = False
                    if target in existing_graph:
                        current_path_length = existing_graph[target].path_length
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
                else:
                    sources.add(target)

    return graph, sources, total_paths_shortened
```

### Memory Feasibility on A100 (40GB)

| Component | Size | Feasible? |
|-----------|------|-----------|
| Full corpus (1.15M × 256 × 128 × 4B) | 150GB | ❌ Too big |
| Chunk (50K × 256 × 128 × 4B) | 6.5GB | ✅ Fits |
| Error matrix (50K × 50K × 4B) | 10GB | ✅ Fits |
| Total per chunk | ~17GB | ✅ Comfortable |
| Number of chunks | 23 | ~30 min total |

### Critical Missing Piece: GPU Transforms

**Current bottleneck**: Transforms applied on CPU, then copied to GPU

```python
# Line 665-670 in emergent_hierarchy.py - SLOW!
transformed_np = lib.apply_transform(  # ⬅️ CPU!
    objects_padded,
    transform['name'],
    transform['amount']
)
transformed_gpu = torch.tensor(transformed_np, device='cuda')  # ⬅️ Copy overhead
```

**Solution**: Implement GPU-native transforms

The codebase has `TensorTransformLibrary` but it's not integrated with the main discovery pipeline.

**Required**: Create GPU transform dispatcher:
```python
def apply_transform_gpu(objects_gpu: torch.Tensor, transform: Dict) -> torch.Tensor:
    """Apply transform directly on GPU."""
    if transform['name'] == 'transpose_semitone':
        # Shift pitch column
        result = objects_gpu.clone()
        result[:, :, 0] += transform['amount']
        return result

    elif transform['name'] == 'time_shift':
        # Roll time axis
        return torch.roll(objects_gpu, shifts=int(transform['amount']), dims=1)

    elif transform['name'] == 'time_scale':
        # Interpolate
        return F.interpolate(objects_gpu.permute(0, 2, 1),
                            scale_factor=transform['amount'],
                            mode='nearest').permute(0, 2, 1)

    elif transform['name'] == 'velocity_scale':
        # Scale velocity column
        result = objects_gpu.clone()
        result[:, :, 2] *= transform['amount']
        return result

    # ... etc for all 29 primitives
```

### Existing GPU Infrastructure

The codebase has these GPU components (but not integrated):

1. **`gpu_discovery_pipeline.py`**: Full GPU pipeline
   - Loads entire corpus to GPU with caching
   - Sparse coding for composition discovery
   - Expected 15-50x speedup

2. **`gpu_sparse_coding.py`**: FISTA optimizer on GPU
   - Finds sparse representations
   - Uses batched operations

3. **`core/tensor_transforms.py`**: GPU transforms (assumed to exist based on imports)

**Problem**: These aren't connected to `emergent_hierarchy.py`'s main discovery loop!

### Recommended Action Plan

#### Phase 1: Enable GPU for Iteration 1 (Already Working) ✅
```bash
python scripts/run_emergent_discovery.py --gpu
```

#### Phase 2: Implement GPU Incremental (2-3 hours of work)
1. Add `_build_derivation_graph_gpu_incremental()` function
2. Implement `apply_transform_gpu()` dispatcher
3. Add chunking logic for 50K objects at a time
4. Update line 483 to enable GPU in incremental mode

#### Phase 3: Test on A100
```bash
python scripts/run_emergent_discovery.py --gpu --max-files 100  # Small test
python scripts/run_emergent_discovery.py --gpu  # Full corpus
```

**Expected performance**:
- Iteration 1 (CPU): 20 minutes
- Iteration 1 (GPU): 3-5 minutes (4-6x speedup)
- Iteration 2+ (CPU): 30-40 minutes each
- Iteration 2+ (GPU): 5-8 minutes each (5-8x speedup)
- **Total**: 90 minutes → 20-30 minutes (3-4x overall)

### Current Limitations

1. **GPU disabled for incremental**: Line 483 blocks GPU for iteration 2+
2. **CPU transforms**: Transform application happens on CPU, not GPU
3. **No chunking**: Would try to load all 1.15M objects at once (OOM)
4. **No path shortening logic**: GPU code doesn't check for shorter paths

### Summary

| Feature | Status | Effort to Complete |
|---------|--------|-------------------|
| GPU iteration 1 (primitives) | ✅ Works | Done |
| GPU iteration 2+ (compositions) | ❌ Disabled | 2-3 hours |
| GPU transform dispatcher | ❌ Missing | 1-2 hours |
| Memory-efficient chunking | ❌ Missing | 1 hour |
| Path shortening on GPU | ❌ Missing | 30 min |
| **Total** | **30% ready** | **4-6 hours** |

### Quick Win

To test GPU immediately for iteration 1:
```bash
python scripts/run_emergent_discovery.py --gpu --max-iterations 1
```

This will use GPU for the initial primitive discovery, then fall back to CPU for iterations 2+.
