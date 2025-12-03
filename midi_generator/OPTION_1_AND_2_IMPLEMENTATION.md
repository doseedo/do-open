# Implementation of Option 1 (Cross-Piece) and Option 2 (MDL Path Shortening)

## Option 2: MDL Path Shortening - COMPLETED ✓

### Changes Made

#### 1. Added `path_length` to Derivation dataclass
**File**: `1_approaches/transform_based/discovery/emergent_hierarchy.py:220-238`
- Added `path_length: int = 1` field to track transform chain length
- Updated `__repr__` to display path length
- This enables MDL (Minimum Description Length) optimization

#### 2. Updated All Derivation Instantiations
**Locations**:
- Line 108: `_process_piece_worker` - sets `path_length=1` for primitives
- Line 189: `_process_piece_incremental_worker` - computes path_length from source
- Line 570: `build_derivation_graph` serial - sets `path_length=1`
- Line 686: `build_derivation_graph` GPU - sets `path_length=1`

#### 3. Modified Incremental Mode to Test ALL Objects
**File**: `emergent_hierarchy.py:425-449`
**Change**: `objects_to_test = objects` instead of `objects_to_test = list(sources)`
**Impact**: Iteration 2+ now tests ALL 1.15M objects for shorter paths, not just 590 sources

#### 4. Implemented Derivation Replacement Logic
**File**: `emergent_hierarchy.py:173-206` (_process_piece_incremental_worker)
```python
if target in existing_graph_piece:
    current_path_length = existing_graph_piece[target].path_length
    if new_path_length < current_path_length:
        # SHORTER PATH FOUND - improves MDL!
        should_add = True
        piece_paths_shortened += 1
```

#### 5. Updated Convergence Criterion
**File**: `scripts/run_emergent_discovery.py:384-392`
**Old**: Stop when sources unchanged
**New**: Stop when sources unchanged AND paths_shortened == 0
**Reasoning**: Even if no new sources derived, compositions might shorten existing paths

#### 6. Added MDL Statistics Tracking
**File**: `scripts/run_emergent_discovery.py:327-351,313-330`
**New stats**:
- `paths_shortened`: Count of derivations replaced with shorter paths
- `avg_path_length`: Average transform chain length (MDL metric)
- `total_description_length`: Sum of all path lengths (total MDL cost)

**Output added**:
```
Paths shortened (MDL improvement): 1234
Average path length: 1.82
```

### Expected Results After Option 2

```
Iteration 1: avg_path_length = 2.3
Iteration 2: avg_path_length = 1.8, paths_shortened = 45,000
Iteration 3: avg_path_length = 1.7, paths_shortened = 12,000
Iteration 4: avg_path_length = 1.65, paths_shortened = 0
→ Converged (no more shortcuts)
```

### Computational Cost

- Iteration 1: ~20-25 minutes (1.15M × 29 primitives)
- Iteration 2+: ~30-40 minutes each (1.15M × 100 compositions)
  - Previous: 590 × 100 = 59K tests (~3 minutes)
  - New: 1.15M × 100 = 115M tests (~30 minutes)
- Trade-off: 10x slower but finds optimal compression

## Option 1: Cross-Piece Discovery - PARTIALLY COMPLETED

### Status

#### ✓ Completed
1. **--cross-piece flag**: Already exists in `run_emergent_discovery.py:181`
2. **Flag plumbing**: Passes `same_piece_only=not args.cross_piece` correctly
3. **Framework support**: Code already handles cross-piece when `same_piece_only=False`

#### ⚠ Needs Optimization (Not Yet Implemented)

The naive cross-piece implementation would be O(N²) with N=1.15M objects = **1.3 trillion comparisons** (IMPOSSIBLE).

**Required optimizations**:

### 1. Signature-Based Filtering (TO BE IMPLEMENTED)

**Goal**: Reduce candidate pairs from 1.3 trillion to ~17 million

**Approach**: Compute lightweight signatures for each object:
```python
def compute_signature(obj: MusicalObject) -> dict:
    """Compute fast similarity signature"""
    return {
        'mean_pitch': np.mean(obj.tensor[:, 0]),  # Average MIDI note
        'pitch_range': np.ptp(obj.tensor[:, 0]),  # Max - min
        'note_density': np.sum(obj.tensor[:, 2] > 0) / len(obj.tensor),  # Active notes
        'rhythm_hash': hash(tuple(obj.tensor[:, 1]))  # Timing pattern
    }
```

**Filtering**:
```python
# Only test transforms if signatures similar
if abs(sig1['mean_pitch'] - sig2['mean_pitch']) < 12:  # Within octave
    if abs(sig1['pitch_range'] - sig2['pitch_range']) < 24:
        if abs(sig1['note_density'] - sig2['note_density']) < 0.3:
            # Candidate pair - test transforms
```

### 2. Sources-Only Cross-Piece (RECOMMENDED FIRST STEP)

**Simpler approach**: Only search cross-piece for the 590 SOURCES
- 590 sources × 1.15M candidates × 29 transforms = 19.7M tests
- Tractable with parallelization (~20-30 minutes)
- Likely finds most cross-piece patterns

**Implementation**:
```python
if same_piece_only == False:
    if target in sources:
        # Search across ALL pieces for this source
        candidates = all_objects_global
    else:
        # Regular within-piece for derived objects
        candidates = objects_in_same_piece
```

### 3. Output Format Enhancement

When cross-piece derivation found:
```
CrossPieceDerive(piece_47→piece_892) ∘ transpose(3)
```

Indicates piece 892 contains material from piece 47 with transform.

### Cross-Piece Expected Results

**Before** (same_piece_only=True):
- 590 sources (unique patterns within their pieces)

**After** (same_piece_only=False with filtering):
- ~150-300 sources (some patterns found across pieces!)
- New discoveries:
  - "Standard ii-V-I lick" appears in 847 pieces
  - "Piece 47 bridge = Piece 203 intro + T(5)"
  - Shared arranger vocabulary

### Performance Impact

| Mode | Comparisons | Time | Memory |
|------|------------|------|---------|
| Within-piece only | 1.15M × 29 | 20 min | 16GB |
| Cross-piece (naive) | 1.3 trillion | IMPOSSIBLE | - |
| Cross-piece (sources only) | 590 × 1.15M × 29 | 30 min | 20GB |
| Cross-piece (signature filter) | ~100M | 60 min | 25GB |

## Summary

### ✅ Fully Implemented
- **Option 2 (MDL Path Shortening)**: Complete
  - Path length tracking
  - Replacement logic
  - Convergence updates
  - Statistics tracking

### ⚠ Partially Implemented
- **Option 1 (Cross-Piece)**: Flag exists, optimization needed
  - --cross-piece flag works
  - But needs signature filtering or sources-only mode for performance

## Next Steps

To complete Option 1, implement ONE of:

1. **Sources-only cross-piece** (easiest):
   - Modify candidate selection to search globally when target is a source
   - 30-minute addition to runtime

2. **Signature-based filtering** (more complex):
   - Add `compute_signature()` function
   - Build signature index
   - Filter candidates before transform testing
   - 1-hour addition to runtime

## Testing

Run with both options:
```bash
# Option 2 only (MDL path shortening)
python scripts/run_emergent_discovery.py

# Option 1 + 2 (cross-piece + MDL)
python scripts/run_emergent_discovery.py --cross-piece

# Small test
python scripts/run_emergent_discovery.py --max-files 10 --cross-piece
```
