# Feature Extractor Comparison & Recommendation

## 📊 All Available Extractors

Based on analysis of your codebase (sorted by modification time):

### 1. **EnhancedFeatureExtractor** (220D) - ✅ MOST RECENT
**File:** `midi_generator/feature_selection/enhanced_feature_extractor.py`
**Modified:** Nov 21 19:02 (TODAY - most recent)
**Status:** ✅ **RECOMMENDED FOR V2.0 TRAINING**

**What it does:**
- Wraps `OptimizedFeatureExtractor` (200D)
- Adds +20D velocity features
- Total: **220D features**

**Architecture:**
```python
EnhancedFeatureExtractor
  └─> OptimizedFeatureExtractor (200D)
        └─> DeepFeatureExtractor (1150D) → select 200 features
  └─> VelocityAnalysis (+20D)
```

**Features:**
- **Base 200D:** Selected from 1150 deep features (harmony, melody, rhythm, dynamics, texture, structure, orchestration)
- **Velocity 20D:** Per-channel velocity statistics (mean, std, min, max, median × 4 channels)

**Dependencies:**
- Requires: `mido` (for velocity extraction)
- Requires: DeepFeatureExtractor working (for base 200D)

**Use with:**
```python
from midi_generator.feature_selection.enhanced_feature_extractor import (
    EnhancedFeatureExtractor,
    NormalizedFeatureExtractor
)

# Load with feature selection
extractor = EnhancedFeatureExtractor.from_selection_file('selected_features_200_actual.json')

# Wrap with normalization (CRITICAL!)
normalized = NormalizedFeatureExtractor(extractor)
normalized.fit(training_files)

features = normalized.extract_batch(training_files)  # Returns (n_files, 220)
```

---

### 2. **RichMultitrackFeatureExtractor** (600D) - 🆕 ALTERNATIVE
**File:** `midi_generator/feature_selection/rich_feature_extractor.py`
**Modified:** Nov 21 19:00 (TODAY)
**Status:** 🔄 **ALTERNATIVE (if mido issues)**

**What it does:**
- Comprehensive multitrack analysis
- Uses `pretty_midi` instead of `mido`
- Total: **600D features**

**Architecture:**
- **Global 200D:** Full-file statistics (wraps DeepFeatureExtractor if available)
- **Per-track 200D:** 8 tracks × 25D (role, density, register, articulation)
- **Temporal 100D:** 4 sections × 25D (evolution over time)
- **Orchestration 100D:** Arrangement quality and balance

**Dependencies:**
- Requires: `pretty_midi` (different from mido)
- Requires: `scipy`

**Use with:**
```python
from midi_generator.feature_selection.rich_feature_extractor import RichMultitrackFeatureExtractor

extractor = RichMultitrackFeatureExtractor()
features = extractor.extract('song.mid')  # Returns (600,)
```

**Note:** Would require changing encoder config to `input_dim=600` instead of 220.

---

### 3. **OptimizedFeatureExtractor** (200D) - 🔧 BASE COMPONENT
**File:** `midi_generator/feature_selection/optimized_feature_extractor.py`
**Modified:** Nov 21 00:10
**Status:** Used internally by EnhancedFeatureExtractor

**What it does:**
- Wraps DeepFeatureExtractor (1150D)
- Selects only 200 features based on feature selection
- Used as base for EnhancedFeatureExtractor

**Issue:** If DeepFeatureExtractor fails (missing `mido`), this breaks silently.

---

### 4. **DeepFeatureExtractor** (1150D) - 🏗️ FOUNDATION
**File:** `midi_generator/synthesis/deep_feature_extractor.py`
**Status:** Foundation for OptimizedFeatureExtractor and EnhancedFeatureExtractor

**What it does:**
- Extracts **1150+ comprehensive musical features**
- Harmony (250), Melody (200), Rhythm (250), Dynamics (150), Texture (100), Structure (50), Orchestration (150)

**Critical Dependency:**
```bash
pip install mido scipy
```

**Without mido:** This extractor CANNOT work, causing cascading failures.

---

## 🔍 Your Current Problem: Only 13 Features

Based on your output:
```
Non-zero features: 12-13/200
[13-199] Padding: [0. 0. 0. 0. 0. 0. 0.]... (all zeros)
```

### Root Cause Analysis:

**Most Likely:** `mido` is not installed in your environment

**What happens:**
1. `DeepFeatureExtractor` cannot import (needs `mido`)
2. System falls back to extracting only basic MIDI stats
3. Only 13 simple features extracted: duration, tempo, note count, pitch stats, velocity stats
4. Rest padded with zeros
5. Training fails because 93% of input is noise

**Verification:**
```bash
conda activate ace_step
python -c "import mido; print('✅ mido installed')"
```

If this fails → **that's your problem!**

---

## ✅ Recommended Solution

### For Your v2.0 Training:

**Use: EnhancedFeatureExtractor (220D) + NormalizedFeatureExtractor**

#### Step 1: Install Dependencies
```bash
conda activate ace_step
pip install mido scipy
```

#### Step 2: Verify It Works
```python
from midi_generator.synthesis.deep_feature_extractor import DeepFeatureExtractor

extractor = DeepFeatureExtractor()
print(f"✅ Feature count: {extractor.feature_count}")  # Should be 1150

# Test extraction
features = extractor.extract('test.mid')
print(f"Non-zero: {(features != 0).sum()}")  # Should be 800+, not 13
```

#### Step 3: Use Full Pipeline
```python
from midi_generator.feature_selection.enhanced_feature_extractor import (
    EnhancedFeatureExtractor,
    NormalizedFeatureExtractor
)

# Create enhanced extractor (220D)
base = EnhancedFeatureExtractor.from_selection_file('selected_features_200_actual.json')

# Wrap with normalization (CRITICAL for convergence!)
normalized = NormalizedFeatureExtractor(base)

# Fit on training data
normalized.fit(training_files, sample_size=100)

# Extract normalized features
features = normalized.extract_batch(training_files)

# Verify
print(f"Shape: {features.shape}")           # (n_files, 220)
print(f"Mean: {features.mean():.3f}")        # ~0.0
print(f"Std: {features.std():.3f}")          # ~1.0
print(f"Non-zero: {(features != 0).sum()}")  # Should be >> 13!
```

---

## 🔄 Alternative If Mido Issues Persist

### Use: RichMultitrackFeatureExtractor (600D)

**Advantages:**
- Uses `pretty_midi` instead of `mido`
- More comprehensive (600D vs 220D)
- May have fewer dependency issues

**Disadvantages:**
- Need to update encoder config: `input_dim=220` → `input_dim=600`
- Need to retrain with larger input dimension

**Setup:**
```bash
pip install pretty_midi scipy
```

**Usage:**
```python
from midi_generator.feature_selection.rich_feature_extractor import RichMultitrackFeatureExtractor

extractor = RichMultitrackFeatureExtractor()
features = extractor.extract('song.mid')  # Returns (600,)
```

**Update encoder config:**
```python
# In modular_encoder_factory.py and semantic_encoder.py
input_dim: int = 600  # Was 220
```

---

## 📋 Quick Decision Matrix

| Extractor | Dimensions | Dependencies | Status | Recommendation |
|-----------|------------|--------------|--------|----------------|
| **EnhancedFeatureExtractor** | 220D | mido, scipy | ✅ Latest | **USE THIS** if mido installs |
| **RichMultitrackFeatureExtractor** | 600D | pretty_midi, scipy | 🆕 New | Use if mido fails |
| **OptimizedFeatureExtractor** | 200D | mido, scipy | 🔧 Internal | Don't use directly |
| **DeepFeatureExtractor** | 1150D | mido, scipy | 🏗️ Foundation | Don't use directly |

---

## 🎯 Action Items

1. **Check if mido is installed:**
   ```bash
   conda activate ace_step
   python -c "import mido; print('OK')"
   ```

2. **If not installed:**
   ```bash
   pip install mido scipy
   ```

3. **Verify feature extraction works:**
   ```bash
   python test_deep_feature_extractor.py
   ```

4. **Expected output:**
   ```
   ✅ Extractor initialized
      Total feature count: 1150
   ✅ Extraction complete:
      Features extracted: 1150
      Non-zero features: 800+/1150  (NOT 13!)
   ```

5. **If step 4 succeeds → Use EnhancedFeatureExtractor (220D)**

6. **If step 4 fails → Use RichMultitrackFeatureExtractor (600D)**

---

## 💡 Key Insight

**Your extracted features showing only 13 non-zero values is NOT normal!**

A properly working feature extractor should give:
- **DeepFeatureExtractor:** 800-1000 non-zero features out of 1150
- **EnhancedFeatureExtractor:** 150-200 non-zero features out of 220
- **RichMultitrackFeatureExtractor:** 400-500 non-zero features out of 600

Getting only 13 non-zero features means:
- ❌ DeepFeatureExtractor is broken (missing `mido`)
- ❌ Only basic stats being extracted
- ❌ No real musical information
- ❌ Training cannot converge on noise

**Fix the extractor first, then retrain!**

---

**Bottom Line:**

✅ **MOST UP TO DATE:** `EnhancedFeatureExtractor` (220D)
✅ **REQUIRES:** `mido` and `scipy` installed
✅ **MUST USE:** `NormalizedFeatureExtractor` wrapper for convergence
✅ **FALLBACK:** `RichMultitrackFeatureExtractor` (600D) if mido fails
