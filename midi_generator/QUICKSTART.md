# Emergent Hierarchy Discovery - Quick Start

## TL;DR - Run Full Corpus Discovery

```bash
cd /home/arlo/do-repo/midi_generator
./scripts/run_full_corpus_discovery.sh
```

That's it! The script will:
- Process all 1,731 MIDI files in the corpus
- Run up to 10 iterations until convergence
- Save results to timestamped directory
- Take approximately 23-77 hours to complete

## What You Get

After completion, you'll have:

```
full_corpus_discovery_results/
└── 20251124_070000/
    ├── discovery_full_corpus.log    # Complete execution log
    ├── iteration_1.json             # First iteration results
    ├── iteration_2.json             # Second iteration results
    ├── ...
    └── final_results.json           # Summary with all discovered patterns
```

## Monitor Progress

```bash
# Follow log in real-time
tail -f full_corpus_discovery_results/*/discovery_full_corpus.log

# Check iteration progress
tail -f full_corpus_discovery_results/*/discovery_full_corpus.log | grep "ITERATION"
```

## Quick Tests

### Test on 30 files (~8 minutes per iteration):
```bash
python -u scripts/test_emergent_hierarchy.py --max-files 30 --max-error 0.03 2>&1 | tee test_30.log
```

### Test on 100 files (~45 minutes per iteration):
```bash
python -u scripts/run_emergent_discovery.py --max-files 100 --max-iterations 3 --output-dir ./test_100 2>&1 | tee test_100.log
```

### Evaluate reconstruction quality:
```bash
python -u scripts/evaluate_reconstruction.py --max-files 30 --max-error 0.03 2>&1 | tee reconstruction.log
```

## Performance Summary

Based on 30-file testing:

| Metric | Value |
|--------|-------|
| Derivation Rate | 99.96% |
| Reconstruction Quality | EXCELLENT |
| Mean Reconstruction MSE | 0.000032 |
| Error Amplification | 0.85x (improves!) |
| Average Chain Depth | 49.5 steps |

## Expected Runtime (Full Corpus - 1,731 files)

- **Per iteration**: ~7.7 hours
- **Total (3-10 iterations)**: ~23-77 hours
- **Memory usage**: ~18GB RAM
- **CPU**: Uses all available cores efficiently

## What Gets Discovered

The algorithm discovers:
1. **Derivation graph**: Which segments derive from which via transforms
2. **Compositions**: Frequent sequences of transforms (e.g., "transpose +7 ∘ time_shift +16")
3. **Meta-patterns**: Parameterized structures (e.g., "transpose X where X ∈ {-7, -5, 0, +5, +7}")
4. **Emergent hierarchy**: Structure emerges from relationships, not predefined levels

## Configuration

Default parameters (optimized for full corpus):

```python
--corpus-path ./midi_corpus/big_band   # 1,731 MIDI files
--scales 64 128 256                     # Multi-scale extraction
--max-error 0.03                        # MSE threshold
--max-iterations 10                     # Stop after 10 iterations
--min-composition-frequency 5           # Composition must appear 5+ times
--max-compositions-per-iteration 100    # Add max 100 per iteration
```

## Troubleshooting

**Out of Memory?**
- Reduce scales: `--scales 64 128`
- Increase error threshold: `--max-error 0.05`

**Too Slow?**
- Test with subset first: `--max-files 100`
- Increase error threshold: `--max-error 0.05`

**Not Converging?**
- Increase min frequency: `--min-composition-frequency 10`
- Decrease max compositions: `--max-compositions-per-iteration 50`

## Full Documentation

See [FULL_CORPUS_DISCOVERY.md](FULL_CORPUS_DISCOVERY.md) for complete details.

## Scripts Reference

| Script | Purpose | Typical Runtime |
|--------|---------|----------------|
| `run_full_corpus_discovery.sh` | Full corpus discovery (recommended) | 23-77 hours |
| `run_emergent_discovery.py` | Iterative discovery (Python) | Configurable |
| `test_emergent_hierarchy.py` | Single-pass test | ~8 min (30 files) |
| `evaluate_reconstruction.py` | Quality evaluation | ~5 min (30 files) |
