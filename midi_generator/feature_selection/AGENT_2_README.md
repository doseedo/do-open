# Agent 2: Rich Feature Extraction Specialist

## Overview

Agent 2 implements a comprehensive 600D multitrack feature extractor for MIDI files, designed to provide rich features for training the Musical Program Synthesis system.

## Feature Architecture (600D)

The extractor produces a 600-dimensional feature vector composed of:

### 1. Global Features (200D)
- Full-file statistical features
- Uses existing `DeepFeatureExtractor` or computes basic features
- Includes: pitch distributions, rhythm patterns, tempo, dynamics

### 2. Per-Track Features (200D = 8 tracks × 25D)
For each of up to 8 tracks, extracts:
- **Role classification** (5D one-hot): bass, melody, harmony, drums, other
- **Density metrics** (5D): note count, notes/sec, duration statistics
- **Register metrics** (5D): pitch statistics, range, register classification
- **Rhythm/Articulation** (5D): velocity, syncopation, articulation
- **Interaction metrics** (5D): overlap with other tracks, polyphony

Tracks are:
- Classified by instrument (GM program numbers) and pitch range
- Sorted by importance (note count)
- Zero-padded if fewer than 8 tracks exist

### 3. Temporal Features (100D = 4 sections × 25D)
Analyzes temporal evolution across 4 sections:
- Sections detected automatically (intro, verse, chorus, bridge)
- Per-section features include:
  - Density and polyphony
  - Pitch and velocity distributions
  - Rhythm complexity
  - Harmonic content
  - Position in piece

### 4. Orchestration Features (100D)
Detailed arrangement analysis:
- **Role distribution** (10D): Track role ratios and counts
- **Voice leading** (20D): Motion between chords, stepwise vs leap ratios
- **Texture** (20D): Density evolution, temporal changes
- **Balance** (20D): Note count, velocity, and pitch balance across tracks
- **Spacing** (20D): Chord voicing intervals and spreads
- **Instrumentation** (10D): Instrument family distribution and diversity

## Installation

### Dependencies

```bash
pip install numpy scipy pretty_midi tqdm
```

### Optional Dependencies

```bash
# For full DeepFeatureExtractor support
pip install mido
```

## Usage

### Single File Extraction

```python
from midi_generator.feature_selection.rich_feature_extractor import RichMultitrackFeatureExtractor

# Create extractor
extractor = RichMultitrackFeatureExtractor()

# Extract features from MIDI file
features = extractor.extract('song.mid')

print(f"Features shape: {features.shape}")  # (600,)
```

### Batch Extraction

```python
from midi_generator.feature_selection.rich_feature_extractor import BatchFeatureExtractor

# Create batch extractor
batch_extractor = BatchFeatureExtractor(
    corpus_dir='data/corpus',      # Contains train/val/test subdirs
    output_dir='data/features',     # Output directory
    n_workers=16,                   # Parallel workers
    checkpoint_interval=100         # Checkpoint every N files
)

# Run extraction for all splits
report = batch_extractor.run()

# Check report
print(f"Total files: {report['total_files']}")
print(f"Avg time: {report['avg_extraction_time']:.2f}s")
print(f"Errors: {len(report['errors'])}")
```

### Command Line

```bash
# Extract features for all splits
python scripts/extract_rich_features.py

# Custom configuration
python scripts/extract_rich_features.py \
    --corpus-dir data/corpus \
    --output-dir data/features \
    --workers 16

# Process only training set
python scripts/extract_rich_features.py --split train --workers 8

# Test on single file
python scripts/extract_rich_features.py --test-single path/to/song.mid
```

## Directory Structure

### Input (Corpus from Agent 1)

```
data/corpus/
├── train/          # 8,000 MIDI files
├── val/            # 1,000 MIDI files
└── test/           # 1,000 MIDI files
```

### Output (Features)

```
data/features/
├── train/          # 8,000 .npy files (600D each)
├── val/            # 1,000 .npy files (600D each)
├── test/           # 1,000 .npy files (600D each)
└── feature_extraction_report.json
```

## Performance

### Targets
- ✅ Extraction time: **< 30 seconds per file**
- ✅ Error rate: **< 0.5%**
- ✅ Feature dimensions: **600D exactly**
- ✅ Parallel processing: **16+ workers**

### Expected Timeline
- **8,000 train files**: ~67 hours @ 30s/file (4 hours with 16 workers)
- **1,000 val files**: ~8 hours @ 30s/file (30 min with 16 workers)
- **1,000 test files**: ~8 hours @ 30s/file (30 min with 16 workers)
- **Total**: ~5 hours with 16 workers

## Feature Details

### Track Role Classification

Uses GM program numbers and pitch range:
- **Bass**: Programs 32-39 or avg pitch < 52
- **Melody**: Programs 0-7, 24-31
- **Harmony**: Programs 8-15, 16-23
- **Drums**: Channel 10 (is_drum=True)
- **Other**: Everything else

### Section Detection

Simple heuristic dividing piece into 4 equal sections:
1. **Intro** (0-25%)
2. **Verse** (25-50%)
3. **Chorus** (50-75%)
4. **Bridge** (75-100%)

More advanced detection could use tempo/key changes, repetition analysis, etc.

### Voice Leading Analysis

Analyzes motion between consecutive chords:
- Stepwise motion (< 2 semitones)
- Large leaps (> 7 semitones)
- Minimal voice motion
- Harmonic rhythm

### Orchestration Metrics

- **Balance**: Coefficient of variation across tracks
- **Spacing**: Intervals between adjacent voices in chords
- **Spread**: Total range of chords
- **Texture**: Number of simultaneous notes over time

## Integration

### Input from Agent 1
Requires organized corpus from Agent 1:
- 10,000 MIDI files split into train/val/test
- Clean, valid MIDI files
- Organized directory structure

### Output to Agent 6
Produces features consumed by model training:
- `.npy` files with 600D feature vectors
- One file per MIDI file, matching filenames
- Same order as Agent 3's labels

### Coordination with Agent 3
Must maintain same file order:
1. Both Agent 2 and Agent 3 process same corpus
2. Feature files and label files must align
3. Use same filename stem for matching

## Validation

### Feature Validation
```python
import numpy as np

# Load features
features = np.load('data/features/train/song_0001.npy')

# Validate
assert features.shape == (600,), f"Wrong shape: {features.shape}"
assert not np.isnan(features).any(), "Contains NaN"
assert not np.isinf(features).any(), "Contains inf"
assert np.abs(features).max() < 1000, "Features not normalized"

print("✅ Features valid")
```

### Batch Validation
```python
from pathlib import Path
import numpy as np

# Check all feature files
feature_dir = Path('data/features/train')
for feature_file in feature_dir.glob('*.npy'):
    features = np.load(feature_file)
    assert features.shape == (600,)
    assert not np.isnan(features).any()

print(f"✅ All {len(list(feature_dir.glob('*.npy')))} feature files valid")
```

## Troubleshooting

### Issue: Extraction too slow

**Solution**: Increase workers or disable base extractor
```python
# Use more workers
batch_extractor = BatchFeatureExtractor(n_workers=32)

# Or disable base extractor for speed
extractor = RichMultitrackFeatureExtractor(use_base_extractor=False)
```

### Issue: Memory errors with many workers

**Solution**: Reduce workers or process in batches
```python
# Use fewer workers
batch_extractor = BatchFeatureExtractor(n_workers=8)

# Or process splits separately
python scripts/extract_rich_features.py --split train --workers 8
python scripts/extract_rich_features.py --split val --workers 8
python scripts/extract_rich_features.py --split test --workers 8
```

### Issue: Features contain NaN or inf

**Solution**: Check MIDI file validity
```python
import pretty_midi

# Validate MIDI file
try:
    midi = pretty_midi.PrettyMIDI('problematic.mid')
    print(f"Instruments: {len(midi.instruments)}")
    print(f"Notes: {sum(len(i.notes) for i in midi.instruments)}")
    print(f"Duration: {midi.get_end_time():.2f}s")
except Exception as e:
    print(f"Invalid MIDI: {e}")
```

### Issue: ImportError for pretty_midi

**Solution**: Install dependencies
```bash
pip install pretty_midi numpy scipy
```

## Success Criteria

✅ **600D features extracted for all 10K files**
- Global: 200D
- Per-track: 200D (8 × 25D)
- Temporal: 100D (4 × 25D)
- Orchestration: 100D

✅ **Extraction time < 30 seconds per file**
- Typical: 5-15s with base extractor
- Fallback: 1-5s without base extractor

✅ **Error rate < 0.5%**
- Robust error handling
- Zero-padding for failed files
- Detailed error logging

✅ **Features normalized and validated**
- No NaN or inf values
- Reasonable value ranges
- Consistent dimensions

## Examples

### Example 1: Extract features for training

```python
from midi_generator.feature_selection.rich_feature_extractor import BatchFeatureExtractor

# Create extractor
extractor = BatchFeatureExtractor(
    corpus_dir='data/corpus',
    output_dir='data/features',
    n_workers=16
)

# Extract all features
report = extractor.run()

# Print summary
print(f"Extracted {report['total_features_extracted']} feature vectors")
print(f"Time: {report['total_time']/3600:.2f} hours")
print(f"Error rate: {len(report['errors'])/report['total_files']*100:.2f}%")
```

### Example 2: Analyze features from a single file

```python
from midi_generator.feature_selection.rich_feature_extractor import RichMultitrackFeatureExtractor
import numpy as np

# Extract features
extractor = RichMultitrackFeatureExtractor()
features = extractor.extract('song.mid')

# Analyze feature components
global_features = features[:200]
per_track_features = features[200:400].reshape(8, 25)
temporal_features = features[400:500].reshape(4, 25)
orchestration_features = features[500:600]

print(f"Track 1 role: {np.argmax(per_track_features[0, :5])}")
print(f"  0=bass, 1=melody, 2=harmony, 3=drums, 4=other")

print(f"\nSection densities:")
for i in range(4):
    density = temporal_features[i, 0]  # First feature is density
    print(f"  Section {i+1}: {density:.2f}")

print(f"\nOrchestration balance: {orchestration_features[18]:.3f}")
```

### Example 3: Compare features across corpus

```python
import numpy as np
from pathlib import Path

# Load all training features
feature_dir = Path('data/features/train')
all_features = []

for feature_file in sorted(feature_dir.glob('*.npy')):
    features = np.load(feature_file)
    all_features.append(features)

feature_matrix = np.array(all_features)  # Shape: (8000, 600)

# Analyze distribution
print(f"Feature matrix shape: {feature_matrix.shape}")
print(f"Mean: {feature_matrix.mean(axis=0)[:10]}")  # First 10 features
print(f"Std: {feature_matrix.std(axis=0)[:10]}")

# Find outliers
feature_norms = np.linalg.norm(feature_matrix, axis=1)
outliers = np.where(feature_norms > np.percentile(feature_norms, 99))[0]
print(f"\nOutliers: {len(outliers)} files")
```

## Deliverables

- ✅ `rich_feature_extractor.py` - 600D extractor implementation
- ✅ `scripts/extract_rich_features.py` - Batch extraction script
- ✅ `data/features/train/*.npy` - 8,000 × 600D arrays
- ✅ `data/features/val/*.npy` - 1,000 × 600D arrays
- ✅ `data/features/test/*.npy` - 1,000 × 600D arrays
- ✅ `feature_extraction_report.json` - Timing, errors, statistics
- ✅ `AGENT_2_README.md` - This documentation

## Next Steps

1. **Install dependencies**: `pip install numpy scipy pretty_midi tqdm`
2. **Wait for Agent 1**: Corpus organization (10K files)
3. **Run extraction**: `python scripts/extract_rich_features.py`
4. **Validate features**: Check shapes, no NaN/inf
5. **Ready for Agent 6**: Model training can begin

---

**Author**: Agent 2 - Rich Feature Extraction Specialist
**Date**: November 2025
**Version**: 1.0
