# Semantic Feature Discovery Examples

This directory contains 7 comprehensive examples demonstrating how to use the Semantic Feature Discovery system.

## Quick Start

Run the examples in order for the best learning experience:

```bash
# 1. Basic discovery (run first - takes 4-8 hours)
python 01_basic_discovery.py

# 2. Custom configurations
python 02_custom_config.py

# 3. Visualize discovered features
python 03_feature_visualization.py

# 4. Extract parameters from MIDI
python 04_parameter_extraction.py

# 5. Compare reconstruction quality
python 05_reconstruction_comparison.py

# 6. Validate across corpora
python 06_cross_corpus_validation.py

# 7. Use in generation
python 07_integration_with_generation.py
```

## Example Descriptions

### 01_basic_discovery.py
**Basic Semantic Feature Discovery**

The simplest way to run feature discovery. Uses default configuration and shows complete workflow.

**What it does:**
- Runs full discovery pipeline
- Discovers 20-30 interpretable features
- Generates HTML report
- Saves results for later use

**Prerequisites:**
- MIDI corpus in `data/midi/train/` (500+ files recommended)

**Runtime:** 4-8 hours on GPU

---

### 02_custom_config.py
**Custom Configuration**

Demonstrates different training configurations for various use cases.

**What it shows:**
- Quick development config (1-2 hours)
- High quality config (8-12 hours)
- Fast reconstruction config (5-7 hours)
- Memory-constrained config (8GB GPU)
- CPU-only config (24-48 hours)

**Use this to:**
- Optimize training for your hardware
- Balance speed vs quality
- Customize loss weights
- Adjust regularization

---

### 03_feature_visualization.py
**Feature Visualization**

Creates visualizations to understand discovered features.

**What it creates:**
- Activation distribution plots (histograms for each feature)
- Feature correlation matrix (heatmap)
- Locality profiles (bar charts)

**Prerequisites:**
- Completed discovery (run 01_basic_discovery.py first)

**Output:** PNG visualizations in `output/visualizations/`

---

### 04_parameter_extraction.py
**Parameter Extraction**

Shows how to extract discovered parameters from MIDI files.

**What it does:**
- Extracts parameters from single MIDI file
- Batch extraction from corpus
- Computes parameter statistics
- Saves parameters to JSON

**Use this to:**
- Analyze your MIDI collection
- Build parameter datasets
- Prepare data for generation

**Output:** JSON file with extracted parameters

---

### 05_reconstruction_comparison.py
**Reconstruction Comparison**

Compares reconstruction quality before and after semantic feature discovery.

**What it measures:**
- Baseline gap (50 existing parameters)
- Gap with semantic features (50 + discovered)
- Improvement percentage
- Per-file comparison

**What it creates:**
- Comparison bar charts
- Improvement histogram
- Summary statistics

**Use this to:**
- Quantify improvement from discovered features
- Identify files that benefit most
- Validate discovery quality

---

### 06_cross_corpus_validation.py
**Cross-Corpus Validation**

Validates that discovered features generalize across different musical styles.

**What it tests:**
- Training corpus performance
- Validation corpus performance
- Held-out test corpus
- Different genres (classical, jazz, pop, etc.)

**What it measures:**
- R² score per corpus
- Consistency across corpora
- Generalization quality

**Use this to:**
- Check for overfitting
- Ensure features work on diverse music
- Validate robustness

---

### 07_integration_with_generation.py
**Integration with MIDI Generation**

Demonstrates using discovered parameters in MIDI generation pipeline.

**What it shows:**
- **Style Transfer:** Clone style from existing MIDI
- **Interpolation:** Blend two musical styles
- **Variation:** Generate variations by tweaking parameters

**Examples:**
- Extract jazz style and apply to new generation
- Interpolate between classical and jazz (5 intermediate styles)
- Increase swing, simplify harmony, add complexity

**Use this to:**
- Build style-based generation systems
- Create parameter-driven composition tools
- Generate variations automatically

---

## Prerequisites

### Required

1. **MIDI Corpus:**
   ```bash
   mkdir -p data/midi/train
   mkdir -p data/midi/validation
   mkdir -p data/midi/test
   # Copy MIDI files to these directories
   ```

   **Recommended sizes:**
   - Training: 500-1000 files
   - Validation: 100-200 files
   - Test: 100-200 files

2. **Python Dependencies:**
   ```bash
   pip install torch numpy scipy scikit-learn mido tqdm matplotlib seaborn
   ```

3. **GPU (recommended):**
   - 8GB+ VRAM
   - CUDA support
   - (CPU works but 10x slower)

### Optional

For some examples, you may want:
- Genre-specific corpora (`data/midi/classical/`, `data/midi/jazz/`, etc.)
- Example MIDI files for style transfer
- Pre-trained encoder (from running example 01)

---

## Expected Outputs

After running all examples, you'll have:

```
output/
├── basic_discovery/
│   ├── feature_bank.pkl
│   ├── encoder.pt
│   ├── results.pkl
│   └── report.html
├── visualizations/
│   ├── feature_activations.png
│   ├── feature_correlations.png
│   └── locality_profiles.png
├── extracted_parameters.json
├── reconstruction_comparison/
│   └── reconstruction_comparison.png
├── interpolation/
│   ├── interp_00_alpha_0.00.mid
│   ├── interp_01_alpha_0.25.mid
│   └── ...
└── variations/
    ├── variation_more_swing.mid
    ├── variation_simpler_harmony.mid
    └── variation_more_complex.mid
```

---

## Troubleshooting

### "CUDA out of memory"
- Reduce batch size in config
- Use gradient accumulation
- Try CPU mode

### "No MIDI files found"
- Check corpus directory exists
- Ensure files are .mid format
- Use absolute paths

### "Feature bank not found"
- Run 01_basic_discovery.py first
- Check output directory path
- Verify training completed successfully

### Training takes too long
- Use smaller corpus for testing (200 files)
- Reduce num_epochs
- Use approximate regeneration method
- Enable caching

See `docs/TROUBLESHOOTING.md` for more help.

---

## Next Steps

After completing these examples:

1. **Customize:** Modify configs for your specific use case
2. **Integrate:** Add discovered parameters to your generation pipeline
3. **Experiment:** Try different loss weights, architectures
4. **Extend:** Add custom test patterns for interpretation
5. **Deploy:** Use in production music generation systems

---

## Documentation

For detailed information, see:
- `docs/SEMANTIC_FEATURE_DISCOVERY.md` - Complete guide
- `docs/API_REFERENCE.md` - API documentation
- `docs/TROUBLESHOOTING.md` - Common issues
- `docs/BENCHMARKS.md` - Performance data

---

**Agent 10 - Documentation & Examples**
**November 2025**
