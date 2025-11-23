# Threshold Selection Guide for MDL Discovery

## Quick Start

### Step 1: Run First Iteration to Gather Data
```bash
python scripts/start_discovery_cpu.py \
  --iterations 1 \
  --cores 56 \
  --threshold 0.0001 \
  2>&1 | tee discovery_iteration_1.log
```

### Step 2: Analyze Improvement Distribution
```bash
python scripts/analyze_improvements.py --log discovery_iteration_1.log
```

This will:
- Fit a power law to your improvement distribution
- Calculate how many primitives needed for each quality target (80%, 90%, 95%, 99%)
- Recommend an optimal threshold
- Generate visualization: `improvement_analysis.png`

### Step 3: Run Full Discovery with Recommended Threshold

**Option A: Fixed Threshold (from analysis)**
```bash
# Use the threshold recommended by analyze_improvements.py
python scripts/start_discovery_cpu.py \
  --iterations 50 \
  --cores 56 \
  --threshold <RECOMMENDED_VALUE> \
  --target-quality 0.99
```

**Option B: Adaptive Threshold (Recommended!)**
```bash
python scripts/start_discovery_cpu.py \
  --iterations 50 \
  --cores 56 \
  --adaptive-threshold \
  --target-quality 0.99
```

Adaptive mode automatically:
- Accepts top 5% of candidates each iteration
- Adapts to changing improvement distributions
- Prevents accepting noise when good candidates exist
- Guarantees progress every iteration

---

## Understanding the Math

### Quality Definition
```
Quality = 1 - (final_error / baseline_error)
```

For 99% quality:
```
final_error = 0.01 * baseline_error
cumulative_improvement = 0.99 * baseline_error
```

### Power Law Distribution

Improvements follow Zipf's law:
```
improvement(k) = α / k^β
```

Where:
- `k` = rank of primitive (1st, 2nd, 3rd...)
- `α` = scaling constant (corpus-dependent)
- `β` = power law exponent (0.5-1.5, typically 0.8-1.0 for music)

### Primitives Needed

Cumulative improvement after `n` primitives:
```
Σ(k=1 to n) α/k^β ≥ target_quality * baseline_error
```

For β ≈ 1.0:
```
Σ(k=1 to n) 1/k ≈ ln(n) + 0.577
```

**Example calculation for 99% quality:**
```
0.99 * E₀ = α * Σ(k=1 to n) 1/k
0.99 * E₀ = α * (ln(n) + 0.577)

If α = 0.15 * E₀ (first primitive gives 15% improvement):
0.99 / 0.15 = ln(n) + 0.577
6.6 = ln(n) + 0.577
n ≈ 403 primitives needed
```

### Threshold Calculation

The weakest primitive you need to accept:
```
improvement(n) = α / n^β
threshold = 0.5 * improvement(n)  # Safety margin
```

**Example:**
```
If n=403, α=0.15*E₀, β=1.0:
improvement(403) = 0.15 * E₀ / 403 ≈ 0.00037 * E₀
threshold ≈ 0.0002 * E₀

If baseline_error = 0.5:
threshold ≈ 0.0001
```

---

## Typical Values for Different Quality Targets

| Target | Primitives | Weakest Improvement | Threshold (E₀=0.5) | Time (56 cores) |
|--------|------------|--------------------|--------------------|-----------------|
| 80%    | 30-50      | 0.005 * E₀         | 0.0025             | 1-2 hours       |
| 90%    | 60-100     | 0.0015 * E₀        | 0.00075            | 3-5 hours       |
| 95%    | 100-200    | 0.0005 * E₀        | 0.00025            | 6-10 hours      |
| 99%    | 200-400    | 0.0002 * E₀        | 0.0001             | 12-24 hours     |

---

## Power Law Exponent (β) Impact

The steepness of the improvement distribution dramatically affects feasibility:

| β (steepness) | Primitives for 99% | Weakest Improvement | Feasible?     |
|---------------|--------------------|--------------------|---------------|
| 0.5 (shallow) | ~150               | 0.012 * E₀         | ✅ Easy       |
| 0.8 (moderate)| ~250               | 0.0006 * E₀        | ✅ Possible   |
| 1.0 (steep)   | ~400               | 0.0004 * E₀        | ⚠️ Difficult  |
| 1.2 (very steep)| ~800             | 0.0002 * E₀        | ❌ Impractical|

**Big band jazz likely has β ≈ 0.8-1.0** because:
- Strong harmonic structure → early big gains (transpose, inversion)
- Diverse styles → long tail of rare patterns (syncopation, voicings)

---

## Decision Tree

```
Run iteration 1 with threshold=0.0001
         ↓
Analyze with analyze_improvements.py
         ↓
Check fitted β value
         ↓
    ┌────┴────┐
    ↓         ↓
 β < 1.0    β ≥ 1.0
    ↓         ↓
 99% feasible  Target 90-95%
    ↓         ↓
Use adaptive   Use adaptive
 threshold     threshold
    ↓         ↓
Run 50+ iters  Run 30-50 iters
```

---

## Monitoring Progress

During discovery, watch for:

### Good Signs ✅
- Multiple PIDs appearing (parallelization working)
- Total time 1-3s per composition
- Improvements varying widely (0.0001 - 0.01)
- Candidates found every iteration (10-30)

### Warning Signs ⚠️
- Only one PID (parallelization broken)
- Total time >10s per composition (bottleneck)
- All improvements near zero (threshold too low)
- No candidates found (threshold too high)

### Expected Iteration Pattern
```
Iteration 1:  20-30 new transforms (quality: 0% → 20%)
Iteration 2-5: 10-20 new transforms (quality: 20% → 50%)
Iteration 6-15: 5-10 new transforms (quality: 50% → 80%)
Iteration 16-30: 2-5 new transforms (quality: 80% → 92%)
Iteration 31-50: 0-2 new transforms (quality: 92% → 95%)
```

Quality plateaus are normal! Diminishing returns expected.

---

## Troubleshooting

### Problem: Found 0-2 candidates
**Solutions:**
1. Lower threshold:
   ```bash
   --threshold 0.00001  # 10× lower
   ```

2. Try triplet compositions:
   ```python
   # Edit cpu_discovery_pipeline.py:
   # Generate 3-way compositions instead of pairwise
   for t1, t2, t3 in itertools.combinations(existing_transforms, 3):
       ...
   ```

3. Check if transforms are actually applied:
   ```bash
   grep "WARNING: didn't change corpus" discovery.log
   ```

### Problem: Found 100+ candidates
**Solutions:**
1. Raise threshold:
   ```bash
   --threshold 0.001  # 10× higher
   ```

2. Use adaptive mode (recommended):
   ```bash
   --adaptive-threshold
   ```

### Problem: Negative improvements
**Causes:**
- Transforms introduce artifacts
- Transform library incomplete (missing inverse transforms)
- Bug in error calculation

**Fix:**
Filter out negative improvements in analysis:
```python
positive_improvements = improvements[improvements > 0]
```

---

## Advanced: Custom Threshold Schedules

### Exponential Decay
```python
def compute_threshold(iteration, baseline_error, max_iter=50):
    threshold_start = 0.001 * baseline_error  # Start high
    threshold_end = 0.00005 * baseline_error  # End low

    progress = iteration / max_iter
    threshold = threshold_start * (threshold_end / threshold_start) ** progress

    return threshold
```

### Top-K Selection
```python
def select_top_k(improvements, k=20):
    sorted_imps = sorted(improvements, reverse=True)
    threshold = sorted_imps[min(k, len(sorted_imps)-1)]
    return threshold
```

---

## Summary

**For 99% quality:**
1. Run iteration 1 with threshold `0.0001`
2. Analyze with `analyze_improvements.py`
3. If β < 1.0: Use recommended threshold, run 50 iterations
4. If β ≥ 1.0: Target 90-95% instead, or accept 12-24 hour runtime
5. **Recommended:** Use `--adaptive-threshold` for automatic tuning

**For quick results (90% quality):**
```bash
python scripts/start_discovery_cpu.py \
  --iterations 30 \
  --cores 56 \
  --adaptive-threshold \
  --target-quality 0.90
```

**Time estimate: 2-5 hours, 60-100 primitives discovered**
