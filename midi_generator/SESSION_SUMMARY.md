# Composition Discovery Session Summary

## Goal
Discover compositional structure in MIDI data by finding transform compositions that reduce code length (MDL/sparsity).

## What We Built

### 1. ✅ MDL Calculation with Sparsity Penalty
**File:** `discovery/cpu_discovery_pipeline.py` lines 394-404

Added proper MDL calculation that penalizes both reconstruction error AND sparsity:
```python
baseline_sparsity = np.mean(np.sum(np.abs(_worker_baseline_encodings) > 1e-6, axis=1))
new_sparsity = np.mean(np.sum(np.abs(new_encodings) > 1e-6, axis=1))

SPARSITY_LAMBDA = 0.0001
baseline_cost = baseline_error + SPARSITY_LAMBDA * baseline_sparsity
new_cost = new_error + SPARSITY_LAMBDA * new_sparsity
improvement = baseline_cost - new_cost
```

**Status:** ✅ Working correctly

---

### 2. ✅ FISTA Optimizer (L1-Regularized Sparse Coding)
**File:** `core/fista_optimizer.py`

Implemented Fast Iterative Shrinkage-Thresholding Algorithm:
- Solves: `minimize ||X - D*a||² + λ||a||₁`
- Proper soft-thresholding for L1 penalty
- Normalization for numerical stability
- Fixed gradient sign bug (was doing gradient ascent!)

**Final working parameters:**
- `lambda_sparsity = 0.1`
- `max_iterations = 200`
- Baseline sparsity: 3.5 transforms/piece
- Reconstruction error: 0.00042 (reasonable!)

**Status:** ✅ Technically correct, but doesn't solve the composition discovery problem

---

### 3. ✅ GreedyEncoder with Composition Bonus
**File:** `core/greedy_encoder.py`

Added composition bonus parameter that biases selection toward compositions:
```python
if '_o_' in transform_name:  # Is a composition
    error *= (1.0 - self.composition_bonus)  # 20% error reduction
```

**Status:** ✅ Implemented and ready to use

---

### 4. ✅ Dependency Graph Tracking
**File:** `scripts/analyze_discovery_dependency_graph.py`

Created tool to analyze which compositions enable future discoveries (keystone vs derived).

**Status:** ✅ Ready for use once discovery produces results

---

## The Fundamental Problem Discovered

### Sparse Coding Cannot Prefer Compositions

**Why compositions show `spar_Δ = 0.00`:**

When testing a composition like `transpose_o_invert`:

1. **Baseline encoding (without composition):**
   - FISTA/Greedy chooses: `[0.8 * transpose, 0.5 * invert]`
   - Sparsity: 2 atoms used
   - Reconstruction: X

2. **Candidate encoding (with composition):**
   - FISTA/Greedy chooses: `[0.8 * transpose, 0.5 * invert]` (SAME!)
   - Or if it uses composition: `[0.95 * transpose_o_invert]`
   - Both give same reconstruction, so why prefer composition?

**The issue:** Free optimization (sparse coding) sees them as equivalent because:
- Composition = Sequential application of primitives
- If `transpose + invert` reconstructs well, composition offers no advantage
- L1 penalty penalizes ALL coefficients equally

**Evidence from logs:**
```
[PID 142433] transpose_semitone_o_time_shift: imp=0.000000, spar_Δ=0.00
[PID 142469] time_scale_o_time_scale: imp=-0.000066, spar_Δ=-0.66  ← WORSE!
```

Compositions either don't help (0.00) or actively hurt (-0.66 means MORE sparsity, not less).

---

## What Worked

1. ✅ **FISTA Implementation** - Numerically stable, correct sparse coding
2. ✅ **MDL with Sparsity Penalty** - Proper objective function
3. ✅ **Composition Bonus in Greedy** - Biases toward compositions
4. ✅ **Debugging Methodology** - Systematic bug hunting (gradient sign, normalization)
5. ✅ **Parallel CPU Pipeline** - 56 workers processing efficiently

---

## What Didn't Work

1. ❌ **Free Sparse Coding for Composition Discovery**
   - FISTA and Greedy both fail to prefer compositions
   - Fundamental limitation: they optimize reconstruction + sparsity freely
   - Compositions are redundant with primitives for reconstruction

2. ❌ **Current Evaluation Approach**
   - Testing if `{base + composition}` is better than `{base}` alone
   - Doesn't account for semantic replacement (composition SHOULD replace its components)

---

## The Correct Approach

### Option 1: Constrained Re-Encoding (Recommended)

Test if using a composition REDUCES sparsity when it REPLACES its component primitives:

```python
def evaluate_composition_correctly(composition, piece, primitives):
    """
    Key insight: If composition is used, its primitives should NOT also be used.
    This is a REPLACEMENT operation, not an addition.
    """
    # Baseline: encode with primitives only
    baseline_codes = encode(piece, primitives)
    baseline_sparsity = count_nonzero(baseline_codes)

    # Candidate: encode with expanded dict
    expanded_dict = primitives + [composition]
    candidate_codes = encode(piece, expanded_dict)

    # Check if composition was selected
    if candidate_codes[composition_idx] > threshold:
        # Composition used!
        # Check if this REDUCED total sparsity
        new_sparsity = count_nonzero(candidate_codes)

        # Also check if component primitives are still heavily used
        component_usage = sum(candidate_codes[component_indices])

        if new_sparsity < baseline_sparsity:
            # Success! Composition replaced primitives and reduced code length
            return baseline_sparsity - new_sparsity

    return 0  # No improvement
```

### Option 2: Pattern Matching

Don't use sparse coding at all. Instead:
1. Find pieces that use `[transpose, invert]` together frequently
2. Replace those joint patterns with `transpose_o_invert`
3. Measure compression directly

### Option 3: Hierarchical Encoding

Force hierarchical structure:
1. Level 1: Only primitives allowed
2. Level 2: Only primitives + 2-compositions
3. Compare code length between levels

---

## Key Lessons Learned

1. **Sparse coding optimizes globally** - doesn't respect semantic structure
2. **Compositions need explicit preference** - can't emerge from pure optimization
3. **Lambda tuning is critical** - 0.0001 too small, 0.1 works but still not enough
4. **Numerical stability matters** - normalization, gradient signs, step sizes
5. **Fast iteration is key** - FISTA is 5-10× slower than Greedy

---

## Current State

### Running Process
- **File:** `discovery_fista_gradient_fixed.log`
- **Status:** Testing 196 compositions with FISTA (15-20 min per batch)
- **Results:** All showing `spar_Δ ≈ 0.00` (no improvement)
- **Recommendation:** Kill and switch to Option 1 above

### Code Ready to Use
- ✅ FISTA optimizer (working)
- ✅ Greedy + composition bonus (working)
- ✅ MDL calculation (working)
- ✅ Dependency graph analyzer (ready)

### What Needs to Change
- ❌ Evaluation logic (implement Option 1)
- ❌ Possibly abandon sparse coding entirely for this task

---

## Next Steps

1. **Implement constrained evaluation** (Option 1 above)
2. **OR: Switch to pattern-matching approach** (Option 2)
3. **OR: Use Greedy + high composition bonus** (0.3-0.5) as pragmatic solution
4. **Test on smaller corpus** to verify approach works before full run

---

## Performance Metrics

### FISTA Timing
- Baseline encoding: ~3 minutes (1720 pieces)
- Per composition test: ~900-1500 seconds (15-25 min)
- Total iteration 1: ~20-25 minutes with 56 workers
- Estimated to 90% quality: 6-8 hours

### Greedy Timing (comparison)
- Per composition test: ~60-180 seconds (1-3 min)
- Total iteration 1: ~3-5 minutes
- Estimated to 90% quality: 1-2 hours

**Conclusion:** FISTA is technically correct but impractical for this problem.

---

## Files Modified

1. `discovery/cpu_discovery_pipeline.py` - Added sparsity penalty, FISTA integration
2. `core/fista_optimizer.py` - New file, full FISTA implementation
3. `core/greedy_encoder.py` - Added composition_bonus parameter
4. `scripts/analyze_discovery_dependency_graph.py` - New file, dependency analysis

---

## Recommended Action

**Kill current run and implement Option 1 (constrained re-encoding)** or **switch to Greedy + high composition bonus (0.3-0.5)** for pragmatic solution.

The current approach is mathematically sound but doesn't solve the composition discovery problem due to fundamental limitations of free sparse coding.
