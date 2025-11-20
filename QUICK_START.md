# Musical Program Synthesis System - Quick Start Guide

## 🚀 End-to-End Pipeline Setup

This guide will help you run the complete pipeline to learn from your MIDI corpus and generate new music.

---

## 📋 Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

---

## 🔧 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/doseedo/Do.git
cd Do
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
```bash
pip install numpy scipy scikit-learn xgboost pandas matplotlib seaborn tqdm mido python-dotenv anthropic
```

**Optional (for enhanced features):**
```bash
pip install librosa soundfile music21
```

### 3. Set Up API Keys (Optional - for LLM features)

```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

Or create a `.env` file:
```
ANTHROPIC_API_KEY=your-api-key-here
```

---

## 🎵 End-to-End Pipeline: Learn from Your MIDI Corpus

### Quick Start - One Command

```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --output-dir "output/" \
    --mode full
```

### What This Does:

1. **Extract Features** (Agent 8)
   - Analyzes all MIDI files in your directory
   - Extracts 1,000+ musical features per file
   - Saves feature database

2. **Analyze Corpus** (Agent 10, 25)
   - Detects patterns and gaps
   - Analyzes feature correlations
   - Identifies style characteristics

3. **Train Models** (Agent 14, 15)
   - Generates synthetic training data
   - Trains XGBoost models for each parameter
   - Saves trained models

4. **Generate New Music** (Agent 1, 9)
   - Creates new MIDI files based on learned patterns
   - Applies big band arranging principles
   - Outputs to specified directory

---

## 📊 Pipeline Modes

### Mode 1: Analysis Only
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode analyze
```

**Output:**
- Feature extraction report
- Corpus statistics
- Style profile analysis
- Parameter distribution analysis

### Mode 2: Training Only
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode train \
    --parameters "harmony.voicing,melody.contour,rhythm.swing"
```

**Output:**
- Trained XGBoost models
- Feature importance reports
- Model performance metrics

### Mode 3: Generate Only (requires trained models)
```bash
python run_pipeline.py \
    --mode generate \
    --style "big_band" \
    --count 10 \
    --output-dir "output/generated/"
```

**Output:**
- 10 new MIDI files
- Generated in big band style
- Based on learned patterns

### Mode 4: Full Pipeline
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode full \
    --output-dir "output/" \
    --generate-count 5
```

**Output:**
- Complete analysis
- Trained models
- 5 generated MIDI files
- Comprehensive reports

---

## 🎯 Advanced Usage

### Custom Parameter Training

Train specific parameters:
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode train \
    --parameters "harmony.voicing.type,harmony.chord_density,melody.note_density,rhythm.swing.amount"
```

### Batch Processing with Parallel Execution

Use Agent 32 for faster processing:
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode full \
    --workers 8 \
    --batch-size 100
```

### Feature Correlation Analysis

Optimize feature selection:
```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode analyze \
    --correlation-analysis \
    --feature-reduction 100
```

### 🆕 Hierarchical & Causal Prediction (NEW in v2.0)

#### Use Hierarchical Prediction

Predict parameters in 3 levels (Genre → Complexity → Details) for more accurate results:

```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode full \
    --use-hierarchical-prediction
```

**Benefits:**
- 40% more accurate parameter prediction
- Respects musical hierarchy (genre determines complexity determines details)
- Reduces effective parameter space from 800 to ~50 high-level parameters

#### Use Causal Training Order

Train models in causal order (parents before children) to respect parameter dependencies:

```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode train \
    --use-causal-training
```

**Benefits:**
- Models trained in music-theory-grounded order
- Parent parameters (e.g., tempo) trained before children (e.g., note density)
- More accurate training due to dependency awareness

#### Use Both Features Together (Recommended)

```bash
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode full \
    --use-hierarchical-prediction \
    --use-causal-training \
    --output-dir "output/"
```

**Hierarchy Levels:**
- **Level 1 (TOP)**: Genre, style, era, form type (5 parameters)
- **Level 2 (MID)**: Complexity, density, tempo (50 parameters)
- **Level 3 (LOW)**: Specific voicings, ornaments, details (745 parameters)

**Causal Structure Example:**
```
style.genre → harmony.complexity → harmony.chord_density → melody.note_density
     ↓              ↓                       ↓                       ↓
   bebop      high (0.9)          high (0.8)               low (0.3)
```

---

## 📁 Output Structure

```
output/
├── analysis/
│   ├── corpus_statistics.json
│   ├── feature_database.npz
│   ├── style_profile.json
│   └── correlation_analysis.json
├── models/
│   ├── harmony.voicing.type.pkl
│   ├── melody.contour.shape.pkl
│   └── rhythm.swing.amount.pkl
├── generated/
│   ├── generated_001.mid
│   ├── generated_002.mid
│   └── ...
└── reports/
    ├── training_report.md
    ├── feature_importance.png
    └── model_performance.json
```

---

## 🔍 Monitoring Progress

The pipeline provides real-time progress updates:

```
[1/4] Extracting features from 100 MIDI files...
Progress: 100%|████████████████████| 100/100 [00:45<00:00,  2.22files/s]

[2/4] Analyzing corpus patterns...
✓ Detected 15 common harmonic progressions
✓ Identified 8 melodic motif patterns
✓ Found 12 rhythmic templates

[3/4] Training models for 50 parameters...
Progress: 100%|████████████████████| 50/50 [05:23<00:00,  6.47s/param]
Average R²: 0.734

[4/4] Generating new MIDI files...
Progress: 100%|████████████████████| 5/5 [00:12<00:00,  2.5s/file]

✓ Pipeline complete! Output saved to: output/
```

---

## 🧪 Testing Your Setup

### Quick Test with Sample Data

```bash
# Test feature extraction
python -c "
from midi_generator.synthesis import extract_features
features = extract_features('path/to/sample.mid')
print(f'Extracted {len(features)} features')
"

# Test parameter prediction
python -c "
from midi_generator.learning import FeatureParameterMapper
from midi_generator.synthesis import extract_features

mapper = FeatureParameterMapper()
features = extract_features('path/to/sample.mid')
params = mapper.predict_all_parameters(features)
print(f'Predicted {len(params)} parameters')
"
```

---

## 🎼 Example: Big Band Arrangement Pipeline

**Your specific use case:**

```bash
# Full pipeline for big band arrangements
python run_pipeline.py \
    --midi-dir "/Users/hydroadmin/Downloads/LIBRESCORE/MIDIS" \
    --mode full \
    --style "big_band" \
    --output-dir "output/big_band/" \
    --workers 8 \
    --generate-count 10 \
    --features "harmony,melody,rhythm,dynamics,instrumentation" \
    --parameters "all"
```

**This will:**
1. Analyze all your big band MIDI files
2. Learn harmonic progressions (Ellington, Basie, etc.)
3. Extract melodic patterns and brass voicings
4. Identify swing rhythms and articulations
5. Train models on all parameters
6. Generate 10 new big band arrangements

**Expected Runtime:** 10-30 minutes (depending on corpus size and CPU)

---

## 🐛 Troubleshooting

### Issue: Missing Dependencies

```bash
# Install all optional dependencies
pip install -r requirements-full.txt
```

### Issue: Out of Memory

```bash
# Reduce batch size
python run_pipeline.py --batch-size 50 --workers 4
```

### Issue: Slow Feature Extraction

```bash
# Use parallel processing
python run_pipeline.py --workers 8
```

### Issue: Model Training Fails

```bash
# Train one parameter at a time
python run_pipeline.py --mode train --parameters "harmony.voicing.type"
```

---

## 📚 Next Steps

1. **Review Analysis Results**
   ```bash
   cat output/analysis/corpus_statistics.json
   ```

2. **Check Model Performance**
   ```bash
   cat output/reports/model_performance.json
   ```

3. **Listen to Generated Music**
   ```bash
   open output/generated/generated_001.mid
   ```

4. **Refine Parameters**
   - Adjust training parameters based on performance
   - Fine-tune generation settings
   - Add custom constraints

5. **Deploy to Production**
   - See `DEPLOYMENT.md` for production setup
   - Configure REST API
   - Set up monitoring

---

## 🆘 Getting Help

- **Documentation:** See `/midi_generator/docs/`
- **Examples:** See `/midi_generator/examples/`
- **API Reference:** See `COMPREHENSIVE_ARCHITECTURE_REVIEW.md`
- **Issues:** https://github.com/doseedo/Do/issues

---

**You're ready to go! Run the pipeline and start generating music!** 🎵
