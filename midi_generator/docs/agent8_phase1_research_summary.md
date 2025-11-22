# Agent 8: Data Pipeline & Preprocessing - Phase 1 Research Summary

**Author:** Agent 8 - Data Pipeline & Preprocessing Optimization
**Date:** 2025-11-22
**Status:** Phase 1 Complete - Research & Understanding

## Executive Summary

I have thoroughly researched the existing data pipeline architecture in the `midi_generator` codebase. The system has **substantial existing infrastructure** for data loading, augmentation, caching, and preprocessing. This document summarizes my findings and outlines how to build on top of this existing code rather than writing from scratch.

## 1. Existing Dataset Classes

### 1.1 GapDataset (Agent 4)
**Location:** `midi_generator/learning/gap_dataset.py`

**Purpose:** Training data for semantic feature discovery through reconstruction gaps

**Key Features:**
- **Input:** MIDI files → (200D features, 50 parameters)
- **Output:** Reconstruction gaps for training
- **Caching:** GapCache with 5-10 GB capacity
- **Batch Loading:** Standard PyTorch DataLoader integration
- **Normalization:** FeatureNormalizer (z-score standardization)
- **Validation:** MusicalCoherenceValidator checks MIDI quality

**Architecture:**
```python
class GapDataset(Dataset):
    def __init__(
        self,
        midi_files: List[Path],
        gap_analyzer: GapAnalyzer,
        cache: Optional[GapCache] = None,
        precompute: bool = True,
        normalize_features: bool = True,
        verbose: bool = False
    )

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.FloatTensor,      # 200D
            'gaps': torch.FloatTensor,          # 200D gaps
            'parameters_flat': torch.FloatTensor,  # 50D params
            'file_id': str
        }
```

### 1.2 MIDILabeledDataset (Agent 3)
**Location:** `midi_generator/learning/dataset_utils.py`

**Purpose:** Labeled dataset for hierarchical parameter learning

**Key Features:**
- **Hierarchical Labels:** Level 1 (global), Level 2 (universal), Level 3 (genre-specific)
- **Dataset Splitting:** Train/val/test with stratified sampling
- **Export Formats:** JSON, CSV, HDF5, pickle
- **Validation:** Built-in dataset validation and statistics

**Architecture:**
```python
class MIDILabeledDataset(Dataset):
    def __getitem__(self, idx) -> Tuple[Any, Dict[str, Any]]:
        return (features, labels)  # Labels are hierarchical dicts
```

### 1.3 HierarchicalMIDIDataset (Agent 6)
**Location:** `midi_generator/training/hierarchical_mtl/data/dataset.py`

**Purpose:** Multi-task learning dataset for 50 hierarchical parameters

**Key Features:**
- **750-file corpus** with complete parameter labels
- **Feature Normalization:** Standardization fitted on training set
- **Data Augmentation:** Noise + scaling augmentation
- **Stratified Sampling:** Genre-balanced sampling option
- **Supports:** Level 1 (8 params), Level 2 (20 params), Level 3 (22 params)

**Architecture:**
```python
class HierarchicalMIDIDataset(Dataset):
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'features': torch.Tensor,  # 200D
            'level1': Dict[str, torch.Tensor],
            'level2': Dict[str, torch.Tensor],
            'level3': Dict[str, torch.Tensor],
            'genre': str,
            'file_id': str
        }
```

## 2. Existing Augmentation Systems

### 2.1 Genre-Specific Augmentation (Agent 7)
**Location:** `midi_generator/multi_genre/augmentation.py`

**Augmentation Types:**
1. **PitchTransposition** - Transpose ±N semitones
2. **TempoScaling** - Scale tempo 0.85-1.15x
3. **VelocityPerturbation** - Gaussian noise on velocities
4. **TimingJitter** - Humanization via timing variation
5. **HarmonicSubstitution** - Jazz chord substitutions (placeholder)

**Genre Configurations:**
- **Jazz:** High variability (±5 semitones, ±15% tempo)
- **Classical:** Conservative (±3 semitones, ±5% tempo)
- **Electronic:** Minimal timing jitter (quantized)
- **Rock/Pop:** Moderate variability

**Usage:**
```python
pipeline = GenreAugmentationPipeline('jazz')
variations = pipeline.augment(midi_data, num_variations=3)
```

### 2.2 Simple Feature Augmentation
**Location:** `midi_generator/training/hierarchical_mtl/data/dataset.py`

```python
class DataAugmenter:
    def __call__(self, features: np.ndarray) -> np.ndarray:
        # Add Gaussian noise
        features = features + np.random.randn(*features.shape) * self.noise_std
        # Random scaling
        scale = np.random.uniform(self.scale_range)
        features = features * scale
        return features
```

## 3. Existing Caching System

### 3.1 GapCache (Agent 4)
**Location:** `midi_generator/learning/gap_dataset.py`

**Features:**
- **SHA256 hashing** for cache keys
- **LRU eviction** when cache exceeds max size
- **Size:** 5-10 GB (configurable)
- **Metadata tracking:** Hit/miss counts, timestamps
- **Persistence:** Pickle-based storage

**Architecture:**
```python
class GapCache:
    def __init__(self, cache_dir: Path, max_size_gb: float = 10.0):
        self.cache_dir = cache_dir
        self.max_size_bytes = int(max_size_gb * 1024**3)

    def get(self, midi_file: Path) -> Optional[ReconstructionGap]:
        # SHA256 hash → cache lookup

    def put(self, midi_file: Path, gap: ReconstructionGap):
        # Cache with automatic eviction

    def get_stats(self) -> Dict:
        return {
            'hit_rate': ...,
            'total_size_gb': ...,
            'entries': ...
        }
```

## 4. Existing Normalization

### 4.1 FeatureNormalizer (Multiple Locations)
**Pattern found in:**
- `hierarchical_mtl/data/dataset.py`
- `feature_selection/optimized_feature_extractor.py`
- `gap_dataset.py`

**Methods:**
1. **Standardization (z-score):**
   ```python
   normalized = (features - mean) / (std + 1e-8)
   ```

2. **MinMax:**
   ```python
   normalized = (features - min) / (max - min + 1e-8)
   ```

**Best Practice Found:**
- Fit normalizer on **training set only**
- Apply same normalization to val/test sets
- Store normalization stats for inference

## 5. Existing Data Generation

### 5.1 SyntheticTrainingDataGenerator (Agent 14)
**Location:** `midi_generator/training/synthetic_data_generator.py`

**Key Features:**
- **Latin Hypercube Sampling** for even parameter space coverage
- **MusicalCoherenceValidator** - validates generated MIDI quality
- **Diverse Parameter Sampling** - prevents overfitting
- **Cross-validation utilities** - k-fold, stratified splits
- **Multiple export formats** - NumPy, PyTorch, HDF5, CSV

**Quality Checks:**
- Has notes (not empty)
- Reasonable length (5-120 seconds)
- Valid pitch range (21-108)
- Appropriate velocities (20-110 average)
- Rhythmic coherence
- Harmonic coherence

## 6. Existing MIDI I/O Patterns

### 6.1 MIDI Loading
**Libraries Used:**
- `mido` - Primary MIDI library
- `pretty_midi` - Alternative (found in some files)

**Common Pattern:**
```python
import mido
midi = mido.MidiFile(str(midi_path))

# Extract notes
for track in midi.tracks:
    for msg in track:
        if msg.type == 'note_on' and msg.velocity > 0:
            # Process note
```

### 6.2 MIDI Validation
**Pattern from MusicalCoherenceValidator:**
```python
def validate_coherence(self, midi) -> float:
    checks = {
        'has_notes': 0-1,
        'reasonable_length': 0-1,
        'pitch_range_ok': 0-1,
        'velocities_ok': 0-1,
        'rhythmic_variation': 0-1,
        'harmonic_coherence': 0-1
    }
    return sum(checks.values()) / len(checks)
```

## 7. Existing Batching & DataLoader Patterns

### 7.1 Standard PyTorch DataLoader Usage
**Pattern found in:** `hierarchical_mtl/data/dataset.py`

```python
def create_dataloaders(...) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Training with optional weighted sampling
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=train_sampler,      # Weighted for genre balance
        shuffle=(train_sampler is None),
        num_workers=4,
        pin_memory=True,
        drop_last=True              # Drop incomplete batches
    )

    # Validation/Test (no augmentation, no shuffling)
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
```

### 7.2 Weighted Random Sampling (for genre balance)
```python
# Compute sample weights
weights = []
for sample in train_dataset.samples:
    genre = sample['labels']['level1'].get('genre.primary', 'unknown')
    weight = 1.0 / genre_dist.get(genre, 1)  # Inverse frequency
    weights.append(weight)

train_sampler = WeightedRandomSampler(
    weights=weights,
    num_samples=len(weights),
    replacement=True
)
```

## 8. Data Flow Architecture

### 8.1 Current End-to-End Flow

```
MIDI File (input)
    ↓
[MIDI Loader] (mido.MidiFile)
    ↓
[Feature Extractor] (200D or 1000D)
    ↓
[Normalization] (z-score on training stats)
    ↓
[Optional Augmentation] (genre-specific pipeline)
    ↓
[PyTorch Dataset] (__getitem__ returns tensors)
    ↓
[DataLoader] (batching, shuffling, multiprocessing)
    ↓
Training/Inference
```

### 8.2 Proposed End-to-End Flow for Agent 1-7 System

```
MIDI File (raw input)
    ↓
[MIDI Loader + Validation] (mido, coherence check)
    ↓
[Optional Caching] (GapCache-style, SHA256 keyed)
    ↓
[Feature Extraction] (reuse existing extractors)
    ↓
[Normalization] (fit on train, apply to val/test)
    ↓
[Augmentation Pipeline] (transposition, tempo, velocity, timing)
    ↓
[PyTorch Dataset]
    ↓
    ├─ [Encoder Path] → DNA (300D)
    │
    └─ [Decoder Path] ← DNA → MIDI (new component)
    ↓
[DataLoader] (efficient batching, variable-length handling)
    ↓
End-to-End Training Loop
```

## 9. Key Insights & Recommendations

### 9.1 What to Reuse (DO NOT Rebuild)

✅ **Reuse These Components:**
1. **GapCache** - Proven 5-10 GB caching system
2. **FeatureNormalizer** - Standardization/MinMax normalization
3. **MusicalCoherenceValidator** - MIDI quality validation
4. **GenreAugmentationPipeline** - Genre-specific augmentation
5. **DatasetSplitter** - Train/val/test stratified splitting
6. **WeightedRandomSampler** - Genre-balanced sampling
7. **MIDI I/O patterns** - mido loading/saving patterns

### 9.2 What to Enhance

🔧 **Enhance These Components:**
1. **Batching for Variable-Length MIDI:**
   - Add custom `collate_fn` for padding
   - Implement sequence packing for efficiency

2. **Caching for End-to-End Training:**
   - Extend GapCache to cache (MIDI → DNA) transformations
   - Cache intermediate representations (features, DNA)

3. **Augmentation for DNA Training:**
   - Build on GenreAugmentationPipeline
   - Add MIDI-to-DNA-to-MIDI augmentation consistency checks

4. **Data Validation:**
   - Extend MusicalCoherenceValidator for DNA reconstruction quality
   - Add reconstruction metrics (pitch accuracy, timing similarity)

### 9.3 What to Build New

🆕 **New Components Needed:**
1. **MIDIReconstructionDataset** - For end-to-end MIDI → DNA → MIDI
2. **Custom collate_fn** - For variable-length MIDI sequences
3. **DNA Caching** - Cache encoded DNA representations
4. **Reconstruction Metrics** - Pitch/timing/harmony similarity

## 10. File Organization Proposal

### 10.1 Existing Structure
```
midi_generator/
├── learning/
│   ├── gap_dataset.py              # GapDataset, GapCache
│   ├── dataset_utils.py            # MIDILabeledDataset, splitting
│   └── dataset_statistics.py
├── multi_genre/
│   └── augmentation.py             # GenreAugmentationPipeline
├── training/
│   ├── hierarchical_mtl/data/
│   │   └── dataset.py              # HierarchicalMIDIDataset
│   └── synthetic_data_generator.py  # Latin Hypercube sampling
└── feature_selection/
    └── optimized_feature_extractor.py  # FeatureNormalizer
```

### 10.2 Proposed Additions for Agent 8
```
midi_generator/
├── data/                           # NEW: Agent 8's contributions
│   ├── __init__.py
│   ├── midi_reconstruction_dataset.py   # End-to-end dataset
│   ├── variable_length_collate.py       # Custom collate functions
│   ├── dna_cache.py                     # DNA representation cache
│   └── reconstruction_metrics.py        # MIDI similarity metrics
├── learning/
│   ├── gap_dataset.py              # EXISTING - reuse as-is
│   ├── dataset_utils.py            # EXISTING - reuse as-is
│   └── dataset_statistics.py       # EXISTING - may extend
├── multi_genre/
│   └── augmentation.py             # EXISTING - reuse & extend
└── training/
    └── hierarchical_mtl/data/
        └── dataset.py              # EXISTING - reuse patterns
```

## 11. Performance Targets (Based on Existing Code)

### 11.1 Current Performance Benchmarks
- **GapCache hit rate:** 60-80% (from code comments)
- **Feature extraction:** ~100 files/second (from agent docs)
- **MIDI loading:** No explicit benchmark, but uses mido (fast)
- **Augmentation:** Real-time (on-the-fly during training)

### 11.2 Agent 8 Targets (from Master Prompt)
- **Load >100 MIDI files/second** ✅ Achievable with caching
- **Augmentation adds <10% overhead** ✅ Current system is real-time
- **Caching reduces load time >5x** ✅ GapCache already proven
- **Memory usage reasonable** - Need to benchmark DNA caching

## 12. Next Steps: Phase 2 Design

### 12.1 Immediate Actions
1. ✅ **Complete Phase 1 Research** (DONE)
2. 📝 **Design Phase 2:** Build on existing components
   - Design `MIDIReconstructionDataset` using GapDataset patterns
   - Design `DNACache` extending GapCache architecture
   - Design variable-length collate functions
   - Design reconstruction metrics

### 12.2 Integration Points with Other Agents
- **Agent 1 (Decoder):** Provide MIDI → DNA → MIDI data flow
- **Agent 2 (Differentiable MIDI):** Integrate soft representations
- **Agent 3 (DNA Expansion):** Support 300D DNA in dataset
- **Agent 5 (Training Loop):** Provide efficient DataLoaders
- **Agent 6 (Semantic Discovery):** Provide variation generation utilities

## 13. Conclusion

The existing codebase has **extensive, high-quality data infrastructure**. Agent 8's role is to:

1. **Leverage existing components** (caching, augmentation, normalization)
2. **Extend for end-to-end training** (MIDI → DNA → MIDI)
3. **Optimize for variable-length sequences** (custom collate functions)
4. **Ensure efficient batching** (reuse existing DataLoader patterns)

**Critical Principle:** Build incrementally on top of existing code rather than reimplementing from scratch. The foundation is solid; we just need to extend it for the new decoder-based architecture.

---

**Phase 1 Status:** ✅ **COMPLETE**
**Next:** Phase 2 Design (Days 4-7)
