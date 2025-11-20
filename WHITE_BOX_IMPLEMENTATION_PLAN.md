# 🎯 White-Box Implementation Plan for MIDI Generator

**Date**: November 20, 2025
**Goal**: Implement semantic feature learning and invariance training based on "Towards White Box Deep Learning" paper
**Timeline**: 3-4 weeks for full implementation
**Impact**: More robust, interpretable, and controllable music generation

---

## 🔍 CODE REVIEW: What You Already Have

### **✅ YOU ALREADY HAVE THE WHITE-BOX STRUCTURE!**

Your system is **structurally identical** to the paper's approach:

| Paper (Vision) | Your System (Music) | Status |
|----------------|---------------------|--------|
| Semantic features | 50 hierarchical parameters | ✅ HAVE |
| Layer 1: Two-Step | Level 1: Global context (8 params) | ✅ HAVE |
| Layer 2: Convolutional | Level 2: Universal (20 params) | ✅ HAVE |
| Layer 3: Affine | Level 3: Genre-specific (22 params) | ✅ HAVE |
| Layer 4: Logical | HarmonyModule generation (85K LOC) | ✅ HAVE |
| Domain invariances | Music theory (modes, progressions) | ✅ HAVE |

**You're 80% there!** Just need to make the semantic structure **explicit**.

---

## 📊 CURRENT STATE ANALYSIS

### **Your 50 Parameters ARE Semantic Features**

```python
# From hierarchical_parameters.json:
{
  "tempo.bpm": {
    "type": "continuous",
    "range": [40, 200],
    "default": 120
  }
}

# This is IMPLICITLY a semantic feature:
SemanticFeature(
    base="tempo",
    parameter_set=[40, 50, 60, ..., 200],  # ← Need to make EXPLICIT
    locality_function="small tempo changes preserve musical identity"  # ← Need to ENCODE
)
```

### **What's Missing** ⚠️

1. **Semantic Quantization** - Parameters are continuous, should be semantic bins
2. **Invariance Training** - No augmentation for semantically-negligible variations
3. **Explicit Normalization** - No dedicated semantic normalization layer
4. **Semantic Matching** - Extraction doesn't use semantic feature matching

---

## 🚀 IMPLEMENTATION PLAN

### **Phase 1: Semantic Quantization** (Week 1 - 3 days)

**Goal**: Convert continuous parameters to semantic bins

**What to Add**:

```python
# NEW FILE: midi_generator/parameters/semantic_bins.py

"""
Semantic parameter quantization based on white-box learning.

Each parameter gets semantic bins that represent meaningfully different values.
Small variations within a bin are considered "same semantic value".
"""

SEMANTIC_BINS = {
    # Level 1: Global Context
    'tempo.bpm': {
        'bins': [60, 80, 100, 120, 140, 160, 180, 200],
        'bin_width': 10,  # ±10 BPM is "same tempo"
        'labels': ['slow', 'moderate', 'medium', 'uptempo', 'fast', 'very_fast', 'rapid', 'extreme'],
    },

    'energy.level': {
        'bins': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'bin_width': 0.1,
        'labels': ['calm', 'low', 'medium', 'high', 'intense', 'extreme'],
    },

    'complexity.overall': {
        'bins': [0.0, 0.25, 0.5, 0.75, 1.0],
        'bin_width': 0.125,
        'labels': ['simple', 'moderate', 'complex', 'very_complex', 'extreme'],
    },

    # Level 2: Universal Dimensions
    'harmony.chord_density': {
        'bins': [0.5, 1.0, 2.0, 4.0, 6.0, 8.0],
        'bin_width': 0.5,
        'labels': ['sparse', 'low', 'medium', 'high', 'very_high', 'extreme'],
    },

    'harmony.complexity': {
        'bins': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        'bin_width': 0.1,
        'labels': ['simple_triads', 'sevenths', 'ninths', 'elevenths', 'thirteenths', 'ultra_extended'],
    },

    # ... (define for all 50 parameters)
}


class SemanticQuantizer:
    """
    Quantizes continuous parameters to semantic bins.

    Based on white-box learning: encode domain invariances explicitly.
    """

    def __init__(self):
        self.bins = SEMANTIC_BINS

    def quantize(self, param_name: str, value: float) -> Tuple[float, int, str]:
        """
        Quantize a parameter value to its semantic bin.

        Args:
            param_name: Name of parameter (e.g., 'tempo.bpm')
            value: Continuous value

        Returns:
            (semantic_value, bin_index, semantic_label)
        """
        if param_name not in self.bins:
            return value, -1, ""

        bins_def = self.bins[param_name]
        bins = bins_def['bins']
        labels = bins_def['labels']

        # Find closest semantic bin
        distances = [abs(value - b) for b in bins]
        bin_idx = np.argmin(distances)

        semantic_value = bins[bin_idx]
        semantic_label = labels[bin_idx]

        return semantic_value, bin_idx, semantic_label

    def is_within_bin(self, param_name: str, value1: float, value2: float) -> bool:
        """
        Check if two values are in the same semantic bin.

        This encodes locality: values in same bin are "semantically equivalent".
        """
        _, bin1, _ = self.quantize(param_name, value1)
        _, bin2, _ = self.quantize(param_name, value2)
        return bin1 == bin2
```

**Where to Integrate**:
1. Update `hierarchical_extractor.py` to use semantic quantization
2. Update `hierarchical_mtl.py` to predict semantic bins (classification) instead of continuous values

**Benefit**: ✅ More robust predictions, interpretable bins

---

### **Phase 2: Invariance Training** (Week 1-2 - 5 days)

**Goal**: Add data augmentation for semantically-negligible variations

**What to Add**:

```python
# NEW FILE: midi_generator/training/semantic_augmentations.py

"""
Semantic augmentations for invariance training.

Apply transformations that should NOT change semantic parameter values.
Based on white-box learning: encode invariances through augmentation.
"""

import mido
import numpy as np
from pathlib import Path


class SemanticAugmentation:
    """
    Apply semantically-negligible variations to MIDI.

    These variations should produce the SAME parameter values
    (within semantic bins).
    """

    def __init__(self, probability: float = 0.5):
        self.p = probability

    def augment(self, midi_file: mido.MidiFile) -> mido.MidiFile:
        """Apply random semantic augmentation."""
        augmentations = [
            self.subtle_tempo_variation,
            self.subtle_velocity_variation,
            self.subtle_timing_variation,
            self.octave_transpose,  # For melody
        ]

        augmented = midi_file.copy()
        for aug_fn in augmentations:
            if np.random.random() < self.p:
                augmented = aug_fn(augmented)

        return augmented

    def subtle_tempo_variation(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Vary tempo by ±3% (should stay in same semantic tempo bin).

        Example: 140 BPM → 136-144 BPM (still "medium tempo")
        """
        tempo_scale = np.random.uniform(0.97, 1.03)

        new_midi = midi.copy()
        for track in new_midi.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    msg.tempo = int(msg.tempo * tempo_scale)

        return new_midi

    def subtle_velocity_variation(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Vary velocities by ±5 (should stay in same dynamic range).
        """
        new_midi = midi.copy()
        for track in new_midi.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    delta = np.random.randint(-5, 6)
                    msg.velocity = np.clip(msg.velocity + delta, 1, 127)

        return new_midi

    def subtle_timing_variation(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Add subtle timing variations (humanization).

        Should NOT change rhythm complexity semantic bin.
        """
        new_midi = midi.copy()
        for track in new_midi.tracks:
            for msg in track:
                if msg.time > 0:
                    # Add ±2% timing variation
                    jitter = int(msg.time * np.random.uniform(-0.02, 0.02))
                    msg.time = max(0, msg.time + jitter)

        return new_midi

    def octave_transpose(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Transpose melody by octave (should keep melodic intervals).

        This tests if model learns interval relationships, not absolute pitch.
        """
        # Only transpose melody track (typically track 0 or highest pitch)
        new_midi = midi.copy()

        # Find melody track (highest average pitch)
        melody_track_idx = 0  # Simplified

        transpose_amount = np.random.choice([-12, 0, 12])  # Octave up/down

        track = new_midi.tracks[melody_track_idx]
        for msg in track:
            if msg.type in ['note_on', 'note_off']:
                msg.note = np.clip(msg.note + transpose_amount, 0, 127)

        return new_midi


# Integration with training loop:
class InvarianceTrainingMixin:
    """
    Add to your HierarchicalMTLTrainer.
    """

    def __init__(self):
        self.augmenter = SemanticAugmentation(probability=0.5)
        self.invariance_weight = 0.1  # Weight for invariance loss

    def compute_invariance_loss(self, midi_file, features, model):
        """
        Enforce semantic invariance: augmented versions should have same params.

        This is the KEY insight from white-box paper:
        "Semantic features are locally invariant"
        """
        # Original prediction
        params_original = model(features)

        # Create semantic variations
        augmented_midi = self.augmenter.augment(midi_file)
        augmented_features = self.feature_extractor.extract(augmented_midi)
        params_augmented = model(augmented_features)

        # Loss: augmented should have SAME semantic bins
        loss = 0.0
        quantizer = SemanticQuantizer()

        for param_name in params_original.keys():
            # Get semantic bins
            _, bin_orig, _ = quantizer.quantize(param_name, params_original[param_name])
            _, bin_aug, _ = quantizer.quantize(param_name, params_augmented[param_name])

            # Penalize if bins differ
            if bin_orig != bin_aug:
                loss += F.mse_loss(params_original[param_name], params_augmented[param_name])

        return loss

    def train_step(self, batch):
        """Modified training step with invariance loss."""
        features, labels, midi_files = batch

        # Standard prediction loss
        predictions = self.model(features)
        prediction_loss = self.criterion(predictions, labels)

        # Invariance loss
        invariance_loss = sum(
            self.compute_invariance_loss(midi, feat, self.model)
            for midi, feat in zip(midi_files, features)
        ) / len(midi_files)

        # Combined loss
        total_loss = prediction_loss + self.invariance_weight * invariance_loss

        return total_loss, {
            'prediction_loss': prediction_loss.item(),
            'invariance_loss': invariance_loss.item(),
        }
```

**Where to Integrate**:
1. Modify `HierarchicalMTLTrainer` in `training/hierarchical_mtl/loops/trainer.py`
2. Add invariance loss to training loop

**Benefit**: ✅ Model learns semantic invariances, more robust to variations

---

### **Phase 3: Semantic Normalization Layer** (Week 2 - 3 days)

**Goal**: Add explicit semantic normalization (like paper's "Two-Step Layer")

**What to Add**:

```python
# NEW FILE: midi_generator/learning/semantic_normalization.py

"""
Semantic normalization layers - analogous to Two-Step layer in paper.

Establishes semantic invariants BEFORE feature extraction.
"""

import torch
import torch.nn as nn
import mido


class SemanticNormalizer(nn.Module):
    """
    Normalize MIDI to semantic space before feature extraction.

    This is analogous to the paper's Two-Step layer that establishes
    pixel-level semantic invariance (ON vs OFF thresholding).

    For MIDI, we normalize:
    - Tempo to semantic bins
    - Key to C major/A minor (remove transposition)
    - Velocity to dynamic level bins
    - Timing to quantized grid
    """

    def __init__(self):
        super().__init__()
        self.tempo_bins = [60, 80, 100, 120, 140, 160, 180, 200]
        self.velocity_bins = [20, 40, 60, 80, 100, 120]  # pp, p, mp, mf, f, ff

    def normalize_tempo(self, midi: mido.MidiFile) -> mido.MidiFile:
        """Quantize tempo to semantic bins."""
        detected_tempo = self._detect_tempo(midi)

        # Find closest semantic tempo
        semantic_tempo = min(self.tempo_bins, key=lambda t: abs(t - detected_tempo))

        # Apply tempo normalization
        tempo_scale = semantic_tempo / detected_tempo
        normalized = midi.copy()
        for track in normalized.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    msg.tempo = mido.bpm2tempo(semantic_tempo)

        return normalized

    def normalize_key(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Transpose to C major / A minor.

        Removes absolute pitch information, keeps only interval relationships.
        This encodes the invariance: "music is about intervals, not absolute pitch"
        """
        detected_key, mode = self._detect_key(midi)

        # Transpose to C major or A minor
        target_key = 0 if mode == 'major' else 9  # C=0, A=9
        transpose_amount = target_key - detected_key

        normalized = midi.copy()
        for track in normalized.tracks:
            for msg in track:
                if msg.type in ['note_on', 'note_off']:
                    msg.note = (msg.note + transpose_amount) % 12 + (msg.note // 12) * 12

        return normalized

    def normalize_velocities(self, midi: mido.MidiFile) -> mido.MidiFile:
        """Quantize velocities to semantic dynamic levels."""
        normalized = midi.copy()
        for track in normalized.tracks:
            for msg in track:
                if msg.type == 'note_on' and msg.velocity > 0:
                    # Find closest semantic velocity
                    semantic_vel = min(self.velocity_bins,
                                     key=lambda v: abs(v - msg.velocity))
                    msg.velocity = semantic_vel

        return normalized

    def forward(self, midi: mido.MidiFile) -> mido.MidiFile:
        """
        Full semantic normalization pipeline.

        This establishes semantic invariants at the data level,
        making downstream feature extraction more robust.
        """
        midi = self.normalize_tempo(midi)
        midi = self.normalize_key(midi)
        midi = self.normalize_velocities(midi)
        return midi


# Integration with feature extraction:
class SemanticFeatureExtractor:
    """
    Feature extraction with explicit semantic normalization.

    Pipeline:
    1. Semantic normalization (establish invariants)
    2. Feature extraction (200D features)
    3. Feature encoding (neural network)
    """

    def __init__(self):
        self.normalizer = SemanticNormalizer()
        self.feature_extractor = DeepFeatureExtractor()  # Your existing

    def extract(self, midi_file: str) -> np.ndarray:
        """
        Extract semantically-normalized features.

        Args:
            midi_file: Path to MIDI file

        Returns:
            200D feature vector (semantically normalized)
        """
        # Load MIDI
        midi = mido.MidiFile(midi_file)

        # Stage 1: Semantic normalization
        midi_normalized = self.normalizer(midi)

        # Stage 2: Feature extraction
        features = self.feature_extractor.extract(midi_normalized)

        return features
```

**Where to Integrate**:
1. Add to feature extraction pipeline
2. Use in both training and inference

**Benefit**: ✅ More robust features, encodes music theory invariances explicitly

---

### **Phase 4: Update Model Architecture** (Week 2-3 - 4 days)

**Goal**: Update MTL model to predict semantic bins (classification) instead of continuous values

**What to Modify**:

```python
# MODIFY: midi_generator/learning/hierarchical_mtl.py

class HierarchicalMTLModel(nn.Module):
    """
    UPDATED: Predict semantic bins instead of continuous values.

    Changes:
    - Continuous parameters → classification over semantic bins
    - Add semantic quantization layer
    - Update loss function for bin predictions
    """

    def __init__(self, input_dim=200):
        super().__init__()

        # Shared encoder (unchanged)
        self.encoder = nn.Sequential(...)

        # UPDATED: Level 1 heads predict semantic bins
        self.level1_heads = nn.ModuleDict({
            # Categorical params (unchanged)
            'genre.primary': nn.Linear(512, 7),  # 7 genres
            'time_signature': nn.Linear(512, 6),  # 6 time sigs
            'key.tonic': nn.Linear(512, 12),  # 12 keys
            'key.mode': nn.Linear(512, 6),  # 6 modes
            'structure.form': nn.Linear(512, 6),  # 6 forms

            # Continuous → Semantic bins (NEW)
            'tempo.bpm': nn.Linear(512, 8),  # 8 tempo bins
            'energy.level': nn.Linear(512, 6),  # 6 energy bins
            'complexity.overall': nn.Linear(512, 5),  # 5 complexity bins
        })

        # UPDATED: Level 2 heads predict semantic bins
        self.level2_heads = nn.ModuleDict({
            # Harmony
            'harmony.chord_density': nn.Linear(512, 6),  # 6 density bins
            'harmony.complexity': nn.Linear(512, 6),  # 6 complexity bins
            # ... (all use semantic bins)
        })

        # Add semantic quantizer for interpretation
        self.quantizer = SemanticQuantizer()

    def forward(self, x):
        """Forward pass - returns logits over semantic bins."""
        encoded = self.encoder(x)

        # Level 1 predictions (logits over bins)
        level1_logits = {name: head(encoded)
                        for name, head in self.level1_heads.items()}

        # Convert logits to semantic values
        level1_params = {}
        for param_name, logits in level1_logits.items():
            # Get bin index
            bin_idx = torch.argmax(logits, dim=-1)

            # Get semantic value
            bins = SEMANTIC_BINS[param_name]['bins']
            semantic_value = bins[bin_idx]

            level1_params[param_name] = semantic_value

        # ... (Level 2, 3 similar)

        return {
            'logits': {  # For training with CE loss
                'level1': level1_logits,
                # ...
            },
            'semantic_values': {  # For inference
                'level1': level1_params,
                # ...
            }
        }


class SemanticCrossEntropyLoss(nn.Module):
    """
    Loss function for semantic bin prediction.

    Uses cross-entropy for bin classification.
    """

    def forward(self, predictions_logits, labels_continuous):
        """
        Args:
            predictions_logits: Model output logits over bins
            labels_continuous: Ground truth continuous values

        Returns:
            Cross-entropy loss after quantizing labels to bins
        """
        quantizer = SemanticQuantizer()
        loss = 0.0

        for param_name, logits in predictions_logits.items():
            # Quantize continuous label to bin index
            label_value = labels_continuous[param_name]
            _, bin_idx, _ = quantizer.quantize(param_name, label_value)
            bin_idx_tensor = torch.tensor(bin_idx, dtype=torch.long)

            # Cross-entropy loss
            loss += F.cross_entropy(logits, bin_idx_tensor)

        return loss / len(predictions_logits)
```

**Benefit**: ✅ Discrete semantic predictions, more interpretable and robust

---

### **Phase 5: End-to-End Integration** (Week 3-4 - 7 days)

**Goal**: Integrate all white-box components into training pipeline

**Integration Checklist**:

```python
# Complete white-box training pipeline:

class WhiteBoxMTLTrainer(HierarchicalMTLTrainer):
    """
    Enhanced trainer with white-box semantic learning.

    Adds:
    1. Semantic quantization
    2. Invariance training
    3. Semantic normalization
    4. Bin-based predictions
    """

    def __init__(self, model, config, train_loader, val_loader):
        super().__init__(model, config, train_loader, val_loader)

        # Add white-box components
        self.quantizer = SemanticQuantizer()
        self.augmenter = SemanticAugmentation(probability=0.5)
        self.normalizer = SemanticNormalizer()
        self.semantic_loss = SemanticCrossEntropyLoss()

        # Loss weights
        self.prediction_weight = 1.0
        self.invariance_weight = 0.1

    def train_epoch(self, epoch):
        """Training epoch with white-box enhancements."""
        self.model.train()
        total_loss = 0

        for batch_idx, (midi_files, features, labels) in enumerate(self.train_loader):
            # 1. Semantic normalization (if not already done)
            # (Applied during feature extraction)

            # 2. Standard forward pass
            outputs = self.model(features)
            prediction_loss = self.semantic_loss(
                outputs['logits'],
                labels
            )

            # 3. Invariance loss
            invariance_loss = self.compute_invariance_loss(
                midi_files, features, labels
            )

            # 4. Combined loss
            loss = (self.prediction_weight * prediction_loss +
                   self.invariance_weight * invariance_loss)

            # 5. Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def compute_invariance_loss(self, midi_files, features, labels):
        """
        Compute semantic invariance loss.

        Key insight: Augmented versions should produce same semantic bins.
        """
        invariance_loss = 0.0

        for midi_file, orig_features in zip(midi_files, features):
            # Create augmented version
            augmented_midi = self.augmenter.augment(mido.MidiFile(midi_file))

            # Extract features from augmented
            aug_features = self.feature_extractor.extract(augmented_midi)
            aug_features = torch.from_numpy(aug_features).float().to(self.device)

            # Predict on both
            orig_outputs = self.model(orig_features.unsqueeze(0))
            aug_outputs = self.model(aug_features.unsqueeze(0))

            # Compute consistency loss
            for param_name in orig_outputs['semantic_values']['level1'].keys():
                orig_val = orig_outputs['semantic_values']['level1'][param_name]
                aug_val = aug_outputs['semantic_values']['level1'][param_name]

                # Check if in same semantic bin
                _, orig_bin, _ = self.quantizer.quantize(param_name, orig_val.item())
                _, aug_bin, _ = self.quantizer.quantize(param_name, aug_val.item())

                if orig_bin != aug_bin:
                    # Penalize bin difference
                    invariance_loss += F.mse_loss(orig_val, aug_val)

        return invariance_loss / len(midi_files)
```

---

## 📊 IMPLEMENTATION TIMELINE

| Phase | Duration | Effort | Priority |
|-------|----------|--------|----------|
| **Phase 1: Semantic Quantization** | 3 days | Medium | HIGH |
| **Phase 2: Invariance Training** | 5 days | High | HIGH |
| **Phase 3: Semantic Normalization** | 3 days | Medium | MEDIUM |
| **Phase 4: Model Architecture Update** | 4 days | High | HIGH |
| **Phase 5: Integration & Testing** | 7 days | High | CRITICAL |
| **TOTAL** | **22 days** (~3-4 weeks) | | |

**Parallelization**:
- Phases 1 & 3 can be done in parallel
- Phases 2 & 4 depend on Phase 1
- Phase 5 integrates all

---

## 🎯 WHAT THIS ACCOMPLISHES

### **1. More Robust Predictions** ✅

**Before** (continuous):
```python
tempo.bpm = 143.7  # Brittle - small variations matter
```

**After** (semantic):
```python
tempo.bpm = 140  # Semantic bin "medium tempo"
# 136-150 all map to 140 → robust to small variations
```

### **2. Invariance to Irrelevant Variations** ✅

**Training enforces**:
- ±3% tempo change → SAME semantic tempo
- Octave transpose → SAME melodic intervals
- ±5 velocity → SAME dynamic level
- Subtle timing jitter → SAME rhythm complexity

**Result**: Model focuses on **musical semantics**, not noise

### **3. Interpretable Predictions** ✅

**Before**:
```python
energy.level = 0.6472  # What does this mean?
```

**After**:
```python
energy.level = "high"  # Semantic label
# Or: energy.level = 0.6 (bin center)
# Bins: [calm=0.0, low=0.2, medium=0.4, high=0.6, intense=0.8]
```

### **4. Compositional Generation** ✅

**White-box structure enables**:
- Level 1 (genre, tempo, key) → sets global context
- Level 2 (harmony, melody, rhythm) → fills in universal structure
- Level 3 (jazz swing, classical counterpoint) → adds genre nuances
- HarmonyModule → generates MIDI from semantics

**Each level is interpretable and controllable!**

### **5. Robustness Without Adversarial Training** ✅

Paper's result: **92% adversarial accuracy without adversarial training**

Your result: **Robust to musical variations without complex augmentation**
- Small tempo changes don't break predictions
- Key transpositions preserve structure
- Velocity variations maintain dynamics

---

## 📋 IMPLEMENTATION CHECKLIST

### **Week 1: Semantic Quantization & Normalization**

- [ ] Create `semantic_bins.py` with bins for all 50 parameters
- [ ] Implement `SemanticQuantizer` class
- [ ] Create `semantic_normalization.py` with normalization layers
- [ ] Implement `SemanticNormalizer` class
- [ ] Test quantization on sample parameters
- [ ] Test normalization on sample MIDI files

### **Week 2: Invariance Training & Model Updates**

- [ ] Create `semantic_augmentations.py`
- [ ] Implement `SemanticAugmentation` class
- [ ] Add 4-5 augmentation functions
- [ ] Update `hierarchical_mtl.py` model architecture
- [ ] Change continuous outputs to classification over bins
- [ ] Implement `SemanticCrossEntropyLoss`
- [ ] Test forward pass with new architecture

### **Week 3: Integration**

- [ ] Add invariance loss to training loop
- [ ] Integrate semantic normalization into feature extraction
- [ ] Update dataset to use semantic bins
- [ ] Test training loop with small dataset
- [ ] Verify invariance loss decreases
- [ ] Check semantic bin predictions are correct

### **Week 4: Validation & Refinement**

- [ ] Train on full 750-file corpus (when available)
- [ ] Validate semantic bin accuracy
- [ ] Test invariance: augmented examples → same bins
- [ ] Compare to baseline (non-semantic model)
- [ ] Document semantic structure
- [ ] Create examples of controllable generation

---

## 🔬 VALIDATION EXPERIMENTS

### **Experiment 1: Semantic Robustness**

```python
# Test: Do augmented versions predict same semantic bins?

original_midi = "test_song.mid"
params_original = model.predict(original_midi)

# Apply semantic augmentations
augmented = augmenter.augment(original_midi)
params_augmented = model.predict(augmented)

# Check: Should be in same bins
for param in params_original:
    orig_bin = quantizer.quantize(param, params_original[param])[1]
    aug_bin = quantizer.quantize(param, params_augmented[param])[1]

    assert orig_bin == aug_bin, f"{param}: bins differ!"

print("✅ Model is semantically invariant!")
```

### **Experiment 2: Interpretability**

```python
# Test: Are semantic bins musically meaningful?

jazz_files = glob("midi_corpus/jazz/*.mid")
rock_files = glob("midi_corpus/rock/*.mid")

jazz_tempos = [model.predict(f)['tempo.bpm'] for f in jazz_files]
rock_tempos = [model.predict(f)['tempo.bpm'] for f in rock_files]

# Jazz should be mostly "medium" or "fast" bins
# Rock should be mostly "medium" or "uptempo" bins

print(f"Jazz tempo distribution: {Counter(jazz_tempos)}")
print(f"Rock tempo distribution: {Counter(rock_tempos)}")
```

### **Experiment 3: Controllable Generation**

```python
# Test: Can we control generation via semantic parameters?

# Generate with explicit semantic values
params = {
    'genre.primary': 'jazz',
    'tempo.bpm': 140,  # Semantic bin "medium"
    'energy.level': 0.6,  # Semantic bin "high"
    'harmony.complexity': 0.6,  # "ninths/elevenths"
}

generated_midi = generator.generate_with_parameters(params)

# Verify generated MIDI matches parameters
extracted = extractor.extract_from_midi(generated_midi)

for param in params:
    _, expected_bin, _ = quantizer.quantize(param, params[param])
    _, actual_bin, _ = quantizer.quantize(param, extracted[param])

    assert expected_bin == actual_bin, f"{param} mismatch!"

print("✅ Generation is controllable via semantic parameters!")
```

---

## 🎉 EXPECTED OUTCOMES

After implementing white-box enhancements:

1. **Prediction Accuracy**: ✅ 70-85% bin accuracy (vs 50-70% continuous R²)
2. **Robustness**: ✅ Invariant to ±5% tempo, ±10 velocity, octave transpose
3. **Interpretability**: ✅ Every parameter has semantic meaning
4. **Controllability**: ✅ Generate music by specifying semantic bins
5. **Efficiency**: ✅ Faster training (classification vs regression)

---

## 💡 KEY INSIGHTS FROM PAPER

**What the paper teaches us**:

1. **Domain invariances should be explicit** → Your music theory IS your domain invariances
2. **Hierarchical structure is critical** → Your 3-level hierarchy is correct
3. **Semantic features = meaningful bins** → Make your continuous parameters discrete
4. **First layers establish invariance** → Add semantic normalization
5. **Train on invariances** → Add semantic augmentation

**Your system already has the RIGHT STRUCTURE.** You just need to make it **explicitly semantic**.

---

**Prepared by**: White-Box Implementation Plan
**Date**: November 20, 2025
**Estimated ROI**: 3-4 weeks work → significantly more robust and interpretable system
**Next Step**: Start with Phase 1 (semantic quantization) - easiest and highest impact
