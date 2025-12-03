# Canonicalization System Deployment

## Date: 2025-11-24

## Summary

Successfully deployed algebraic path canonicalization to eliminate degenerate composition patterns in the emergent hierarchy discovery pipeline.

## Key Changes

### 1. Octave Equivalence Setting - **FIXED FOR BIG BAND**

**Problem:** Initial implementation treated T(±12) as identity (octave equivalence), which is incorrect for big band arranging where octave doublings are musically meaningful (bass playing octave below, etc.).

**Solution:** Changed default `octave_equivalence=False` in `is_identity()` function.

**Impact:**
- ✅ T(12) now recognized as meaningful octave transposition
- ✅ Bass doublings, octave harmonizations preserved
- ✅ Compositions like "T(-12) ∘ velocity_scale(0.7)" kept as 2-step patterns
- ✅ All tests updated and passing

### 2. Parallel CPU Configuration - **LIMITED TO 50 CORES**

**Changes Made:**
- Added `--num-workers` argument to `scripts/run_emergent_discovery.py`
- Updated `discovery/emergent_hierarchy.py` to accept `num_workers` parameter
- Modified `scripts/run_full_corpus_discovery.sh` to use `--num-workers 50`

**Configuration:**
```bash
# System has 60 cores available
# Using 50 cores for discovery (leaves 10 for system)
python scripts/run_emergent_discovery.py --num-workers 50
```

### 3. Deployment Status

**Running Now:**
- Background job ID: 96379b
- Output directory: `./full_corpus_discovery_results/20251124_085033/`
- Corpus: 1,731 MIDI files from big band corpus
- Configuration:
  - Max iterations: 10
  - Max error: 0.03
  - Min composition frequency: 5
  - Scales: 64, 128, 256 timesteps
  - CPU workers: 50 parallel processes

**Expected Improvements Over Previous Run:**
1. **No degenerate patterns** like `time_shift(-16) ∘ T(12) ∘ T(-12) ∘ T(12) ∘ T(-12)...`
2. **Meaningful octave compositions** preserved (e.g., `T(-12)` for bass)
3. **Cycle detection** prevents infinite bidirectional paths
4. **Max path length = 4** instead of 10 (more focused)
5. **Trivial paths filtered** (length ≤ 1 after simplification)

## Architecture Summary

### Transform Algebra (`core/transform_algebra.py`)
- Additive groups: transpose_semitone, time_shift
- Multiplicative groups: time_scale, velocity_scale
- Involutions: inversion, retrograde
- **Octave equivalence: FALSE by default** (big band specific)

### Path Canonicalizer (`core/path_canonicalizer.py`)
- Combines adjacent same-type transforms
- Cancels inverse pairs
- Removes identity transforms
- Stores both canonical and temporal representations

### Discovery Integration (`discovery/emergent_hierarchy.py`)
- Cycle detection with visited node tracking
- Max path depth: 4 (down from 10)
- Parallel processing: 50 workers
- Filters: min_len=2, max_len=4

## Expected Output Format

### Before Canonicalization (from previous run):
```json
{
  "compositions": [
    {
      "path": "time_shift(-16.0) ∘ transpose_semitone(12.0) ∘ transpose_semitone(-12.0) ∘ ...",
      "frequency": 244400
    }
  ]
}
```

### After Canonicalization (expected now):
```json
{
  "compositions": [
    {
      "canonical": "transpose_semitone(7.0) ∘ time_shift(16.0)",
      "frequency": 45000,
      "raw_variants": [...]
    }
  ],
  "temporal_data": {
    "canonical_to_raw": {...},
    "identity_processes": {...}
  }
}
```

## Monitoring Discovery Progress

Check output:
```bash
# View latest output
tail -f ./full_corpus_discovery_results/20251124_085033/discovery_full_corpus.log

# Check current iteration results
ls -lah ./full_corpus_discovery_results/20251124_085033/

# Monitor system resources
htop  # Should see ~50 Python processes
```

## Validation Checklist

Once discovery completes, verify:

- [ ] No degenerate patterns in top 20 compositions
- [ ] Compositions have length 2-4 (not 10+)
- [ ] Octave transpositions (T(±12)) appear in results
- [ ] Compositions are musically interpretable
- [ ] Temporal data includes raw path variants
- [ ] Identity processes tracked separately
- [ ] Derivation rate ~99.5% (similar to before)
- [ ] Iteration convergence achieved

## Files Modified in This Deployment

1. `core/transform_algebra.py` - Changed octave_equivalence default to False
2. `tests/test_canonicalizer.py` - Updated tests for octave-aware behavior
3. `scripts/run_emergent_discovery.py` - Added --num-workers argument
4. `discovery/emergent_hierarchy.py` - Added num_workers parameter propagation
5. `scripts/run_full_corpus_discovery.sh` - Set --num-workers 50

## Comparison with Previous Run

**Previous Run (20251124_073459):**
- Statistics:
  - Total objects: 1,157,202
  - Total derivations: 1,156,612
  - Avg graph depth: 19.74
  - Total compositions: 2,005
- **Problem:** Top compositions were degenerate (T(12) ∘ T(-12) chains)

**Current Run (20251124_085033):**
- **Expected:** Same object/derivation counts
- **Expected:** Much fewer unique compositions (~100-500 canonical forms)
- **Expected:** All compositions musically meaningful
- **Expected:** Avg graph depth reduced (max_depth=4)

## Performance Notes

**CPU Usage:**
- 50 parallel workers processing pieces independently
- Each worker handles one piece at a time
- Load balanced by piece count (1,731 pieces)

**Memory:**
- Similar to previous run (~same object count)
- Slightly more due to temporal data storage
- Est. 20-40 GB RAM for full corpus

**Time Estimate:**
- Previous run: ~2-3 hours per iteration
- Expected similar or slightly faster (reduced path exploration)
- Total: 4-6 hours for convergence

## Next Steps After Completion

1. **Analyze Results:**
   - Compare top compositions with previous run
   - Verify degenerate patterns eliminated
   - Check temporal variants for interesting patterns

2. **Musical Interpretation:**
   - Map compositions to arranging techniques
   - Identify style-specific patterns
   - Correlate with music theory

3. **Iterative Refinement:**
   - Adjust min_composition_frequency if needed
   - Fine-tune max_len based on results
   - Consider adding more primitive transforms

4. **Documentation:**
   - Document discovered compositions
   - Create musical interpretations guide
   - Publish findings

---

**Status:** ✅ Deployed and Running
**Job ID:** 96379b
**Start Time:** 2025-11-24 08:50:33 UTC
**Expected Completion:** ~6 hours (approx. 2025-11-24 14:50 UTC)
