# 🔍 Extraction Format Assessment: Current vs Required

**Assessment Date**: November 20, 2025
**Question**: Is your current extraction format correct for training the hierarchical MTL model?
**Answer**: **❌ NO - Missing critical 200D feature vector**

---

## 📊 CURRENT EXTRACTION FORMAT (What You Have)

### Example from "Les_Feuilles_Mortes" (Autumn Leaves)

```json
{
  "file_id": "Les_Feuilles_Mortes",
  "file_path": "midi_corpus/big_band/Les_Feuilles_Mortes.mid",
  "labels": {
    "level1_global": {
      "tempo.bpm": 140.0,
      "time_signature": "4/4",
      "key.tonic": "F",
      "key.mode": "minor",
      "genre.primary": "electronic",  // ❌ WRONG - should be "jazz"
      "structure.form": "through_composed",
      "energy.level": 0.695,
      "complexity.overall": 0.748
    },
    "level2_universal": {
      "harmony": { /* 6 params */ },
      "melody": { /* 5 params */ },
      "rhythm": { /* 5 params */ },
      "dynamics": { /* 2 params */ },
      "texture": { /* 2 params */ }
      // ✅ Total: 20 params - CORRECT
    },
    "level3_genre_specific": {
      "orchestration": { /* 5 params */ },
      "electronic": { /* 3 params */ }
      // ❌ Total: 8 params - WRONG (should be 22)
    }
  },
  "metadata": { ... }
}
```

### **CRITICAL ISSUES:**

#### ❌ **Issue 1: Missing 200D Feature Vector**
```python
# What you have:
extraction = {
    'features': None,  # ❌ MISSING!
    'labels': { /* parameters only */ }
}

# What you need:
extraction = {
    'features': [200],  # ✅ REQUIRED for neural encoder INPUT
    'parameters': { /* parameters as labels */ }
}
```

**Why critical:** Your neural network trains like this:
```
INPUT: 200D features → Neural Encoder → OUTPUT: 50 parameters
   ↑ MISSING!                                  ✅ You have
```

#### ❌ **Issue 2: Incomplete Level 3 Parameters**
```python
current_level3 = {
    'orchestration': 5,  # Universal params
    'electronic': 3,     # Electronic-specific
    'total': 8          # ❌ Only 8/22!
}

required_level3 = {
    'orchestration': 5,     # Universal (always active)
    'jazz': 4,              # ❌ MISSING
    'classical': 3,         # ❌ MISSING
    'rock': 3,              # ❌ MISSING
    'electronic': 3,        # Has, but missing 1 param
    'hiphop': 2,           # ❌ MISSING
    'latin': 2,            # ❌ MISSING
    'total': 22            # Should be 22!
}
```

#### ❌ **Issue 3: Genre Misclassification**
```json
{
  "file_id": "Les_Feuilles_Mortes",  // Famous JAZZ standard!
  "genre.primary": "electronic"       // ❌ Completely wrong
}
```

"Les Feuilles Mortes" (Autumn Leaves) is one of the most famous **jazz standards** ever written. It should activate jazz-specific parameters, not electronic.

---

## ✅ REQUIRED EXTRACTION FORMAT (What You Need)

### Complete Training Sample Format

```json
{
  "file_id": "Les_Feuilles_Mortes",
  "file_path": "midi_corpus/big_band/Les_Feuilles_Mortes.mid",

  // ✅ NEW: 200D Feature Vector (Neural Encoder INPUT)
  "features": [
    0.234,   // feature 0: harmony_chord_quality_major_triad_ratio
    0.567,   // feature 1: harmony_chord_quality_minor_triad_ratio
    -0.123,  // feature 2: harmony_chord_quality_dominant_seventh_ratio
    // ... 197 more features ...
    0.891    // feature 199: structure_repetition_score
  ],

  // ✅ RENAMED: "labels" → "parameters" (Neural Encoder OUTPUT)
  "parameters": {

    // Level 1: Global Context (8 params) ✅
    "level1_global": {
      "genre.primary": "jazz",           // ✅ CORRECTED
      "tempo.bpm": 140.0,
      "time_signature": "4/4",
      "key.tonic": "F",
      "key.mode": "minor",
      "energy.level": 0.7,
      "complexity.overall": 0.75,
      "structure.form": "AABA"           // ✅ Jazz standard form
    },

    // Level 2: Universal Dimensions (20 params) ✅ Already correct
    "level2_universal": {
      "harmony": {
        "chord_density": 6.0,
        "complexity": 0.8,
        "chromaticism": 0.6,
        "tension": 0.5,
        "voicing_spread": 0.7,
        "progression_predictability": 0.6
      },
      "melody": {
        "note_density": 8.0,
        "range_semitones": 21,
        "contour_smoothness": 0.66,
        "rhythmic_complexity": 0.62,
        "repetition": 0.34
      },
      "rhythm": {
        "subdivision": "sixteenth",
        "syncopation": 0.32,
        "groove_consistency": 0.85,
        "polyrhythm": 0.01,
        "swing_amount": 0.67
      },
      "dynamics": {
        "overall_level": 0.59,
        "range": 0.09
      },
      "texture": {
        "polyphony": 6,
        "density": 7.72
      }
    },

    // Level 3: Genre-Specific (22 params) ✅ ALL GENRES
    "level3_genre_specific": {

      // Universal Orchestration (5 params - always active)
      "orchestration": {
        "instrument_count": 10,
        "register_balance": 0.49,
        "legato_ratio": 0.97,
        "section_contrast": 0.5,
        "repetition_level": 0.5
      },

      // Jazz (4 params) - ✅ ACTIVE (genre = jazz)
      "jazz": {
        "swing_feel": "medium",            // ✅ NEW
        "walking_bass": 0.85,              // ✅ NEW
        "improvisation_ratio": 0.4,        // ✅ NEW
        "bebop_vocabulary": 0.6            // ✅ NEW
      },

      // Classical (3 params) - Set to 0 (not classical)
      "classical": {
        "counterpoint": 0.0,
        "development_density": 0.0,
        "voice_leading_quality": 0.0
      },

      // Rock (3 params) - Set to 0 (not rock)
      "rock": {
        "power_chord_ratio": 0.0,
        "riff_repetition": 0.0,
        "distortion_level": 0.0
      },

      // Electronic (3 params) - Set to 0 (not electronic)
      "electronic": {
        "quantization": 0.0,
        "filter_movement": 0.0,
        "arpeggio_density": 0.0
      },

      // Hip-Hop (2 params) - Set to 0 (not hiphop)
      "hiphop": {
        "sample_based": 0.0,
        "boom_bap_feel": 0.0
      },

      // Latin (2 params) - Set to 0 (not latin)
      "latin": {
        "clave_pattern": "none",
        "montuno_complexity": 0.0
      }

      // Total: 5 + 4 + 3 + 3 + 3 + 2 + 2 = 22 ✅
    }
  },

  "metadata": {
    "total_notes": 1543,
    "duration_seconds": 199.74,
    "extraction_version": "2.0.0"
  }
}
```

---

## 🔧 WHAT YOU ALREADY HAVE (Good News!)

### ✅ You Already Have the Feature Extractor!

```python
# File: midi_generator/synthesis/deep_feature_extractor.py
class DeepFeatureExtractor:
    """Extract 1000+ musical features from MIDI files"""

    def extract(self, midi_file) -> np.ndarray:
        # Returns 1000+ dimensional feature vector
        pass
```

### ✅ You Already Have the Optimized Extractor!

```python
# File: midi_generator/feature_selection/optimized_feature_extractor.py
class OptimizedFeatureExtractor:
    """Extract only the selected 200 features"""

    def __init__(self, selected_features: List[str]):
        # Loads 200 selected features
        self.base_extractor = DeepFeatureExtractor()

    def extract(self, midi_file) -> np.ndarray:
        # Returns 200-dimensional feature vector
        full_features = self.base_extractor.extract(midi_file)  # 1000+ features
        return full_features[self.selected_indices]              # Select 200
```

### ✅ You Have the 200 Selected Features!

```json
// File: midi_generator/feature_selection/output/selected_features_200_template.json
{
  "n_features_selected": 200,
  "selected_features": [
    "harmony_chord_quality_major_triad_ratio",
    "harmony_chord_quality_minor_triad_ratio",
    "harmony_voicing_density_mean",
    "melody_note_density",
    "melody_range_semitones",
    "rhythm_syncopation_probability",
    "dynamics_overall_level",
    "texture_polyphony_mean",
    // ... 192 more features ...
  ]
}
```

---

## 🚨 WHAT YOU NEED TO FIX

### Fix 1: Integrate Feature Extraction into Hierarchical Extractor

**Current:** Your `HierarchicalParameterExtractor` only extracts parameters
**Required:** Also extract 200D feature vector

```python
# File: midi_generator/parameters/hierarchical_extractor.py

class HierarchicalParameterExtractor:
    """
    Extract BOTH features AND parameters from MIDI files.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._load_hierarchical_schema()

        # ✅ ADD: Initialize optimized feature extractor
        from midi_generator.feature_selection.optimized_feature_extractor import OptimizedFeatureExtractor

        # Load selected features
        selected_features_path = Path(__file__).parent.parent / \
            'feature_selection/output/selected_features_200_template.json'
        with open(selected_features_path) as f:
            selected_data = json.load(f)
            selected_features = selected_data['selected_features']

        self.feature_extractor = OptimizedFeatureExtractor(selected_features)

    def extract_from_midi(self, midi_path: str) -> Dict[str, Any]:
        """
        Extract BOTH features (200D) AND parameters (50) from MIDI file.

        Returns:
            {
                'features': np.ndarray (200,),      # ✅ NEW
                'parameters': {                     # ✅ RENAMED from 'labels'
                    'level1_global': {...},
                    'level2_universal': {...},
                    'level3_genre_specific': {...}
                },
                'metadata': {...}
            }
        """
        if self.verbose:
            print(f"Extracting from: {midi_path}")

        # ✅ NEW: Extract 200D feature vector
        features = self.feature_extractor.extract(Path(midi_path))
        assert features.shape == (200,), f"Expected 200 features, got {features.shape}"

        # ✅ EXISTING: Extract parameters
        analysis = self._analyze_midi(midi_path)
        level2 = self._extract_level2(analysis)
        level1 = self._extract_level1(analysis, level2)
        level3 = self._extract_level3(analysis, level1, level2)

        # ✅ VERIFY: 50 total parameters
        level1_count = len(level1)
        level2_count = sum(len(v) for v in level2.values())
        level3_count = sum(len(v) for v in level3.values())

        assert level1_count == 8, f"Level 1: expected 8, got {level1_count}"
        assert level2_count == 20, f"Level 2: expected 20, got {level2_count}"
        assert level3_count == 22, f"Level 3: expected 22, got {level3_count}"

        return {
            'features': features.tolist(),  # ✅ NEW: 200D vector
            'parameters': {                 # ✅ RENAMED from 'labels'
                'level1_global': level1,
                'level2_universal': level2,
                'level3_genre_specific': level3
            },
            'metadata': {
                'file': str(midi_path),
                'extraction_version': '2.1.0',  # Bump version
                'total_notes': analysis.total_notes,
                'duration_seconds': analysis.duration_seconds,
                'feature_count': len(features)  # Should be 200
            }
        }
```

### Fix 2: Extract ALL 22 Level 3 Parameters

**Current:** Only extracts parameters for detected genre (8 params)
**Required:** Extract ALL genre parameters, set to 0 if not applicable (22 params)

```python
def _extract_level3(
    self,
    analysis: MIDIAnalysis,
    level1: Dict,
    level2: Dict
) -> Dict[str, Any]:
    """
    Extract ALL 22 Level 3 parameters.
    Set to 0.0 for non-applicable genres.
    """
    detected_genre = level1['genre.primary']

    level3 = {}

    # Universal Orchestration (5 params - always active)
    level3['orchestration'] = {
        'instrument_count': len(analysis.instrument_programs),
        'register_balance': self._calculate_register_balance(analysis),
        'legato_ratio': self._calculate_legato_ratio(analysis),
        'section_contrast': 0.5,  # Placeholder
        'repetition_level': 0.5   # Placeholder
    }

    # Jazz (4 params) - extract regardless of genre
    level3['jazz'] = {
        'swing_feel': self._detect_swing_feel(analysis) if detected_genre == 'jazz' else 'straight',
        'walking_bass': self._detect_walking_bass(analysis) if detected_genre == 'jazz' else 0.0,
        'improvisation_ratio': self._estimate_improvisation(analysis) if detected_genre == 'jazz' else 0.0,
        'bebop_vocabulary': self._detect_bebop(analysis) if detected_genre == 'jazz' else 0.0
    }

    # Classical (3 params)
    level3['classical'] = {
        'counterpoint': self._detect_counterpoint(analysis) if detected_genre == 'classical' else 0.0,
        'development_density': self._detect_development(analysis) if detected_genre == 'classical' else 0.0,
        'voice_leading_quality': self._evaluate_voice_leading(analysis) if detected_genre == 'classical' else 0.0
    }

    # Rock (3 params)
    level3['rock'] = {
        'power_chord_ratio': self._detect_power_chords(analysis) if detected_genre == 'rock' else 0.0,
        'riff_repetition': self._detect_riffs(analysis) if detected_genre == 'rock' else 0.0,
        'distortion_level': self._estimate_distortion(analysis) if detected_genre == 'rock' else 0.0
    }

    # Electronic (3 params)
    level3['electronic'] = {
        'quantization': self._measure_quantization(analysis) if detected_genre == 'electronic' else 0.0,
        'filter_movement': 0.5 if detected_genre == 'electronic' else 0.0,  # Placeholder
        'arpeggio_density': self._detect_arpeggios(analysis) if detected_genre == 'electronic' else 0.0
    }

    # Hip-Hop (2 params)
    level3['hiphop'] = {
        'sample_based': self._detect_loops(analysis) if detected_genre == 'hiphop' else 0.0,
        'boom_bap_feel': self._detect_boom_bap(analysis) if detected_genre == 'hiphop' else 0.0
    }

    # Latin (2 params)
    level3['latin'] = {
        'clave_pattern': self._detect_clave(analysis) if detected_genre == 'latin' else 'none',
        'montuno_complexity': self._detect_montuno(analysis) if detected_genre == 'latin' else 0.0
    }

    # ✅ VERIFY: Total 22 parameters
    total_params = sum(len(v) for v in level3.values())
    assert total_params == 22, f"Expected 22 Level 3 params, got {total_params}"

    return level3
```

### Fix 3: Improve Genre Classification

**Current:** "Les Feuilles Mortes" classified as "electronic" ❌
**Required:** Classify as "jazz" ✅

```python
def _classify_genre(
    self,
    analysis: MIDIAnalysis,
    level2: Dict
) -> str:
    """
    Improved genre classification.

    Uses Level 2 features to make better decisions.
    """
    harmony = level2['harmony']
    rhythm = level2['rhythm']

    # Jazz indicators
    has_swing = rhythm['swing_amount'] > 0.6
    has_complex_harmony = harmony['complexity'] > 0.7
    has_syncopation = rhythm['syncopation'] > 0.3

    if has_swing and has_complex_harmony:
        return 'jazz'

    # Electronic indicators
    perfect_timing = rhythm['groove_consistency'] > 0.95
    low_swing = rhythm['swing_amount'] < 0.55

    if perfect_timing and low_swing:
        return 'electronic'

    # Rock indicators
    simple_harmony = harmony['complexity'] < 0.4
    high_energy = level2['dynamics']['overall_level'] > 0.7

    if simple_harmony and high_energy:
        return 'rock'

    # Classical indicators
    has_counterpoint = level2['texture']['polyphony'] > 6
    smooth_voice_leading = harmony['voicing_spread'] < 0.4

    if has_counterpoint and smooth_voice_leading:
        return 'classical'

    # Default to most likely genre based on other features
    return self._heuristic_genre_classification(analysis, level2)
```

---

## 📋 TRAINING DATA FORMAT

### What the Neural Network Expects

```python
# During training:
train_loader = DataLoader(dataset, batch_size=32)

for batch in train_loader:
    # batch['features']: torch.Tensor of shape (32, 200)  ← INPUT
    # batch['level1']: Dict of tensors (32, ...)          ← OUTPUT labels
    # batch['level2']: Dict of tensors (32, ...)          ← OUTPUT labels
    # batch['level3']: Dict of tensors (32, ...)          ← OUTPUT labels

    features = batch['features']  # (32, 200)

    # Forward pass
    outputs = model(features)  # Returns predictions for all 50 params

    # Compute loss
    loss = criterion(outputs, batch['level1'], batch['level2'], batch['level3'])
```

### Dataset Class Expectation

```python
# From: midi_generator/training/hierarchical_mtl/data/dataset.py

class HierarchicalMIDIDataset(Dataset):
    """
    Dataset expects JSON with this structure:
    """

    def __init__(self, labeled_dataset_path, features_dir=None):
        # Load labeled_dataset.json
        with open(labeled_dataset_path) as f:
            self.data = json.load(f)

        # Each sample should have:
        # - 'features': [200 floats]  ← REQUIRED
        # - 'parameters': {level1, level2, level3}  ← REQUIRED

    def __getitem__(self, idx):
        sample = self.data[idx]

        # Extract features (200D)
        features = torch.tensor(sample['features'], dtype=torch.float32)

        # Extract parameters (50 total)
        level1 = self._convert_level1(sample['parameters']['level1_global'])
        level2 = self._convert_level2(sample['parameters']['level2_universal'])
        level3 = self._convert_level3(sample['parameters']['level3_genre_specific'])

        return {
            'features': features,
            'level1': level1,
            'level2': level2,
            'level3': level3
        }
```

---

## 🎯 ACTION PLAN

### Step 1: Update Hierarchical Extractor (1-2 hours)

```bash
# Edit: midi_generator/parameters/hierarchical_extractor.py

1. Add OptimizedFeatureExtractor integration
2. Modify extract_from_midi() to return features + parameters
3. Implement complete Level 3 extraction (all 22 params)
4. Improve genre classification logic
5. Add verification asserts (200 features, 50 params)
```

### Step 2: Re-extract Your Corpus (20-35 minutes for 2K files)

```python
# Script: extract_corpus_complete.py

from midi_generator.parameters.hierarchical_extractor import HierarchicalParameterExtractor
from pathlib import Path
import json

extractor = HierarchicalParameterExtractor(verbose=True)
corpus_dir = Path("midi_corpus/big_band/")
labeled_data = []

for midi_file in corpus_dir.glob("**/*.mid"):
    print(f"Processing: {midi_file.name}")

    # Extract features (200D) + parameters (50)
    extraction = extractor.extract_from_midi(str(midi_file))

    # Verify format
    assert 'features' in extraction
    assert len(extraction['features']) == 200
    assert 'parameters' in extraction

    # Add to dataset
    labeled_data.append({
        'file_id': midi_file.stem,
        'file_path': str(midi_file),
        'features': extraction['features'],  # 200D
        'parameters': extraction['parameters'],  # 50 params
        'metadata': extraction['metadata']
    })

# Save
with open("labeled_dataset_complete.json", "w") as f:
    json.dump(labeled_data, f, indent=2)

print(f"✅ Extracted {len(labeled_data)} files")
print(f"✅ Each sample has 200 features + 50 parameters")
```

### Step 3: Verify Data Format (5 minutes)

```python
# verify_dataset.py

import json
import numpy as np

with open("labeled_dataset_complete.json") as f:
    data = json.load(f)

sample = data[0]

# Verify structure
assert 'features' in sample
assert 'parameters' in sample

# Verify feature count
assert len(sample['features']) == 200, f"Expected 200 features, got {len(sample['features'])}"

# Verify parameter counts
params = sample['parameters']
level1_count = len(params['level1_global'])
level2_count = sum(len(v) for v in params['level2_universal'].values())
level3_count = sum(len(v) for v in params['level3_genre_specific'].values())

assert level1_count == 8, f"Level 1: expected 8, got {level1_count}"
assert level2_count == 20, f"Level 2: expected 20, got {level2_count}"
assert level3_count == 22, f"Level 3: expected 22, got {level3_count}"

print("✅ Dataset format verified!")
print(f"✅ Total samples: {len(data)}")
print(f"✅ Features: 200D")
print(f"✅ Parameters: {level1_count + level2_count + level3_count} (8+20+22)")
```

### Step 4: Train! (5-10 minutes with GPU)

```python
# train_big_band.py

from midi_generator.learning.hierarchical_mtl import HierarchicalMTLModel
from midi_generator.training.hierarchical_mtl.data.dataset import create_dataloaders
from midi_generator.training.hierarchical_mtl.config.training_config import get_fast_config
from midi_generator.training.hierarchical_mtl.loops.trainer import HierarchicalMTLTrainer

# Config
config = get_fast_config()
config.data.labeled_dataset_path = "labeled_dataset_complete.json"

# Data (now with 200D features!)
train_loader, val_loader, test_loader = create_dataloaders(
    labeled_dataset_path=config.data.labeled_dataset_path,
    batch_size=32
)

# Model
model = HierarchicalMTLModel(input_dim=200)  # ✅ 200D input

# Trainer
trainer = HierarchicalMTLTrainer(model, config, train_loader, val_loader, test_loader)

# Train!
results = trainer.train()
```

---

## ⚖️ SHOULD YOU IMPLEMENT ADDITIONAL FEATURES?

### **Answer: NO - You already have everything you need!**

**What you already have:**
- ✅ DeepFeatureExtractor (1000+ features)
- ✅ OptimizedFeatureExtractor (200 selected features)
- ✅ Selected features list (200 best features)
- ✅ Hierarchical MTL model architecture
- ✅ Complete training pipeline

**What you need to do:**
- ✅ Integrate existing OptimizedFeatureExtractor into HierarchicalParameterExtractor
- ✅ Fix Level 3 to extract all 22 parameters
- ✅ Improve genre classification
- ✅ Re-extract corpus with correct format

**Time estimate:** 2-4 hours to integrate + 30 min to re-extract corpus

**Bottom line:** **DON'T build new features - integrate what you already have!**

---

## 📊 FINAL COMPARISON

| Component | Current | Required | Status |
|-----------|---------|----------|--------|
| **Features** | ❌ None | ✅ 200D vector | MISSING |
| **Level 1 Params** | ✅ 8 | ✅ 8 | CORRECT |
| **Level 2 Params** | ✅ 20 | ✅ 20 | CORRECT |
| **Level 3 Params** | ❌ 8 | ✅ 22 | INCOMPLETE |
| **Genre Classification** | ❌ Wrong | ✅ Correct | BROKEN |
| **Total Ready** | **40%** | **100%** | **FIX NEEDED** |

---

## ✅ CONCLUSION

**Your current extraction format is NOT ready for training.**

**Critical fixes needed:**
1. ❌ Add 200D feature vector extraction (integrate OptimizedFeatureExtractor)
2. ❌ Extract all 22 Level 3 parameters (currently only 8)
3. ❌ Fix genre classification ("Les Feuilles Mortes" should be jazz)

**Good news:**
- ✅ You already have all the code needed (DeepFeatureExtractor, OptimizedFeatureExtractor)
- ✅ You already have the 200 selected features
- ✅ Level 1 and Level 2 extraction is correct
- ✅ Training pipeline is ready

**Timeline:** 2-4 hours to integrate + 30 min re-extraction = **Ready to train same day!**

**Recommendation:** Fix the integration, re-extract your 2K corpus, then train immediately. Don't build new features - use what you have!

---

**Assessment by**: Claude (Architecture Review)
**Date**: November 20, 2025
**Confidence**: High (verified against training pipeline requirements)
