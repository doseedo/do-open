# Semantic Feature Discovery Troubleshooting Guide

**Version:** 1.0.0
**Last Updated:** November 2025
**Agent:** 10 - Documentation & Examples

---

## Table of Contents

1. [Common Issues](#common-issues)
2. [Installation Problems](#installation-problems)
3. [Training Issues](#training-issues)
4. [Memory Problems](#memory-problems)
5. [Performance Issues](#performance-issues)
6. [Interpretation Problems](#interpretation-problems)
7. [Validation Failures](#validation-failures)
8. [Integration Issues](#integration-issues)
9. [Data Issues](#data-issues)
10. [Debugging Tips](#debugging-tips)

---

## Common Issues

### Issue: "RuntimeError: CUDA out of memory"

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 8.00 GiB total capacity)
```

**Cause:** Batch size too large for available GPU memory.

**Solutions:**

**Option 1: Reduce batch size**
```python
config = TrainingConfig(
    batch_size=16,  # Reduced from 32
    ...
)
```

**Option 2: Use gradient accumulation**
```python
# In trainer, accumulate gradients over multiple batches
config = TrainingConfig(
    batch_size=8,
    gradient_accumulation_steps=4  # Effective batch size = 8*4 = 32
)
```

**Option 3: Use CPU**
```python
config = TrainingConfig(
    device="cpu"  # Slower but works with any RAM
)
```

**Option 4: Use mixed precision training**
```python
# Requires PyTorch 1.6+
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

# In training loop:
with autocast():
    outputs = encoder(x)
    loss = encoder.compute_loss(...)

scaler.scale(loss['total_loss']).backward()
scaler.step(optimizer)
scaler.update()
```

---

### Issue: Training loss not decreasing

**Symptom:**
```
Epoch 1/100  Train loss: 0.5432
Epoch 2/100  Train loss: 0.5429
Epoch 3/100  Train loss: 0.5431
...
Epoch 20/100  Train loss: 0.5427
```

**Possible Causes:**

**1. Learning rate too low**

```python
# Try increasing learning rate
config = TrainingConfig(
    learning_rate=1e-3  # Increased from 1e-4
)
```

**2. Learning rate too high (loss oscillating)**

```python
# Try decreasing learning rate
config = TrainingConfig(
    learning_rate=1e-5  # Decreased from 1e-4
)
```

**3. Locality weight too high**

The model might be focusing too much on locality at the expense of reconstruction.

```python
config = TrainingConfig(
    locality_weight=0.1  # Reduced from 0.5
)
```

**4. Sparsity weight too high**

```python
config = TrainingConfig(
    sparsity_weight=0.001  # Reduced from 0.01
)
```

**5. Bad initialization**

Try reinitializing and training again:

```python
encoder = SemanticFeatureEncoder(...)
# Train again
```

**Diagnostic: Check individual loss components**

```python
# After training, plot individual losses
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(history['reconstruction_loss'])
plt.title('Reconstruction Loss')

plt.subplot(1, 3, 2)
plt.plot(history['locality_loss'])
plt.title('Locality Loss')

plt.subplot(1, 3, 3)
plt.plot(history['sparsity_loss'])
plt.title('Sparsity Loss')

plt.show()
```

If one component is much larger than others, adjust its weight.

---

### Issue: Features not interpretable

**Symptom:**
```
Interpretation Results:
  Automatically interpreted: 8/25
  Require manual review: 17
```

**Solutions:**

**1. Increase locality weight during training**

This makes features more semantically coherent:

```python
config = TrainingConfig(
    locality_weight=0.8  # Increased from 0.5
)
```

**2. Reduce number of features**

Fewer features force each one to be more meaningful:

```python
encoder = SemanticFeatureEncoder(
    num_semantic_features=15  # Reduced from 25
)
```

**3. Increase sparsity**

Sparser features are often more interpretable:

```python
config = TrainingConfig(
    sparsity_weight=0.05  # Increased from 0.01
)
```

**4. Use larger corpus**

More diverse data helps features learn robust concepts:

```
# Use 1000+ MIDI files instead of 500
```

**5. Adjust interpretation threshold**

Lower threshold to accept more interpretations:

```python
results = interpreter.interpret_all_features(
    corpus_dir=corpus,
    threshold=0.4  # Lowered from 0.6
)
```

---

### Issue: Training is very slow

**Symptom:**
Training takes > 12 hours on GPU.

**Solutions:**

**1. Use cached gaps**

Enable aggressive caching:

```python
dataset = GapDataset(
    midi_files,
    cache_dir=Path("cache/gaps"),
    use_cache=True  # Make sure this is True
)
```

**2. Use approximate regeneration**

Much faster than exact:

```python
dataset = GapDataset(
    ...,
    regeneration_method="approximate"  # Instead of "exact"
)
```

**3. Reduce corpus size during development**

Use a subset for initial experiments:

```python
# Only use first 200 files for quick testing
midi_files = list(corpus_dir.glob("*.mid"))[:200]
```

**4. Use more data loading workers**

```python
config = TrainingConfig(
    num_workers=8  # Increased from 4
)
```

**5. Enable pin_memory**

Faster GPU transfer:

```python
config = TrainingConfig(
    pin_memory=True
)
```

---

### Issue: "FileNotFoundError" during gap computation

**Symptom:**
```
FileNotFoundError: [Errno 2] No such file or directory: '/tmp/tmpXYZ.mid'
```

**Cause:** Temporary files being deleted before processing.

**Solution:**

Increase temp file lifetime or use a persistent temp directory:

```python
import tempfile
import os

# Set custom temp directory
custom_temp = Path("tmp/midi_transforms")
custom_temp.mkdir(parents=True, exist_ok=True)

os.environ['TMPDIR'] = str(custom_temp)
```

---

### Issue: Validation failures (features not passing)

**Symptom:**
```
Validation Results:
  Passed: 12/25
  Failed: 13
```

**Solutions:**

**1. Check which validations are failing**

```python
results = validator.validate_all_features(features)

for feat in results['invalid_features']:
    details = results['validation_details'][feat.feature_id]
    print(f"\n{feat.name} failed:")
    for check in details.checks_failed:
        print(f"  - {check}")
```

**2. If "musical_validity" failing:**

Features aren't behaving musically. Increase locality weight during training.

**3. If "redundancy" failing:**

Too many similar features. Reduce num_features or increase diversity:

```python
encoder = SemanticFeatureEncoder(
    num_semantic_features=20  # Reduced from 25
)
```

**4. If "interpretability" failing:**

See "Features not interpretable" section above.

---

## Installation Problems

### Issue: PyTorch not using GPU

**Symptom:**
```python
>>> import torch
>>> torch.cuda.is_available()
False
```

**Solutions:**

**1. Check CUDA installation**

```bash
nvidia-smi
```

If this fails, CUDA drivers aren't installed.

**2. Reinstall PyTorch with CUDA**

```bash
# For CUDA 11.8
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**3. Check PyTorch CUDA version matches system CUDA**

```python
import torch
print(torch.version.cuda)  # Should match nvidia-smi CUDA version
```

---

### Issue: "ModuleNotFoundError: No module named 'midi_generator'"

**Symptom:**
```python
>>> from midi_generator.learning.semantic_discovery_pipeline import SemanticDiscoveryPipeline
ModuleNotFoundError: No module named 'midi_generator'
```

**Solutions:**

**1. Check you're in the right directory**

```bash
pwd
# Should show: /path/to/Do
ls midi_generator
# Should list files
```

**2. Add to Python path**

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
```

**3. Install in development mode**

```bash
pip install -e .
```

---

## Training Issues

### Issue: NaN loss

**Symptom:**
```
Epoch 5/100  Train loss: nan
```

**Causes & Solutions:**

**1. Learning rate too high**

```python
config = TrainingConfig(
    learning_rate=1e-5  # Much smaller
)
```

**2. Exploding gradients**

```python
config = TrainingConfig(
    gradient_clip=0.5  # Smaller clip threshold
)
```

**3. Bad batch**

Skip problematic batches:

```python
# In training loop
if torch.isnan(loss['total_loss']):
    print(f"Skipping batch {batch_idx} (NaN loss)")
    continue
```

---

### Issue: Training stops early (early stopping triggered too soon)

**Symptom:**
```
Early stopping at epoch 15
```

But validation loss was still decreasing.

**Solution:**

Increase patience:

```python
config = TrainingConfig(
    early_stopping_patience=20  # Increased from 10
)
```

---

### Issue: Model overfitting (train loss << val loss)

**Symptom:**
```
Epoch 50/100
  Train loss: 0.0234
  Val loss: 0.1892
```

**Solutions:**

**1. Increase dropout**

```python
encoder = SemanticFeatureEncoder(
    dropout=0.3  # Increased from 0.1
)
```

**2. Increase weight decay**

```python
config = TrainingConfig(
    weight_decay=1e-4  # Increased from 1e-5
)
```

**3. Use more training data**

Collect more MIDI files for training.

**4. Reduce model capacity**

```python
encoder = SemanticFeatureEncoder(
    hidden_dim=256,  # Reduced from 512
    num_semantic_features=15  # Reduced from 25
)
```

---

## Memory Problems

### Issue: System runs out of RAM (not GPU memory)

**Symptom:**
```
MemoryError: Unable to allocate array
```

**Solutions:**

**1. Reduce number of workers**

```python
config = TrainingConfig(
    num_workers=2  # Reduced from 4
)
```

**2. Don't load entire corpus into memory**

Use lazy loading:

```python
# In GapDataset, ensure streaming mode
dataset = GapDataset(
    midi_files,
    preload=False  # Don't load all into memory
)
```

**3. Clear cache periodically**

```python
import gc
import torch

# After each epoch
gc.collect()
torch.cuda.empty_cache()
```

---

### Issue: Gap cache consuming too much disk space

**Symptom:**
```
cache/gaps/  120 GB
```

**Solutions:**

**1. Use compressed cache format**

```python
# Modify GapDataset to use compressed numpy
np.savez_compressed(cache_file, **gap_data)  # Instead of np.savez
```

**2. Clean old cache**

```bash
# Delete cache older than 7 days
find cache/gaps -name "*.npz" -mtime +7 -delete
```

**3. Use smaller cache**

Only cache most frequently used files:

```python
# Cache only if file accessed > 2 times
cache_access_counts = {}
# ... implement access tracking
```

---

## Performance Issues

### Issue: Data loading is the bottleneck

**Symptom:**
GPU utilization < 50% during training.

**Solutions:**

**1. Increase number of workers**

```python
config = TrainingConfig(
    num_workers=8  # Increased from 4
)
```

**2. Prefetch more batches**

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    prefetch_factor=4,  # Prefetch 4 batches per worker
    persistent_workers=True  # Don't restart workers each epoch
)
```

**3. Use SSD for cache**

Move cache directory to SSD:

```python
dataset = GapDataset(
    midi_files,
    cache_dir=Path("/mnt/ssd/cache/gaps")  # Fast SSD path
)
```

---

### Issue: Inference is slow (>1 second per MIDI)

**Symptom:**
Extracting semantic features takes too long.

**Solutions:**

**1. Batch inference**

```python
# Process multiple files at once
midi_files = [Path("song1.mid"), Path("song2.mid"), ...]

# Extract features for all
from midi_generator.parameters.optimized_feature_extractor import (
    OptimizedFeatureExtractor
)

extractor = OptimizedFeatureExtractor()
all_features = []

for midi_file in midi_files:
    features = extractor.extract(midi_file)
    all_features.append(features)

all_features = np.array(all_features)  # (num_files, 200)

# Batch encode
with torch.no_grad():
    x = torch.tensor(all_features, dtype=torch.float32)
    semantic = encoder.encode(x)  # (num_files, K)
```

**2. Use ONNX export**

Export model to ONNX for faster inference:

```python
import torch.onnx

dummy_input = torch.randn(1, 200)
torch.onnx.export(
    encoder,
    dummy_input,
    "semantic_encoder.onnx",
    input_names=['features'],
    output_names=['semantic_features'],
    dynamic_axes={'features': {0: 'batch_size'}}
)

# Use ONNX runtime for inference
import onnxruntime as ort

session = ort.InferenceSession("semantic_encoder.onnx")
semantic_features = session.run(
    None,
    {'features': features_200d.reshape(1, -1).astype(np.float32)}
)[0]
```

**3. Quantize model**

Reduce model precision for faster inference:

```python
# Quantize to int8
quantized_encoder = torch.quantization.quantize_dynamic(
    encoder,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Use quantized model for inference (2-4x faster)
```

---

## Interpretation Problems

### Issue: Feature names are not meaningful

**Symptom:**
```
feature_0_unnamed
feature_1_unnamed
feature_2_unnamed
```

**Solutions:**

**1. Provide custom test patterns**

```python
# Create custom MIDI files representing musical concepts
test_patterns_dir = Path("test_patterns")

# Add files like:
# test_patterns/heavy_swing.mid
# test_patterns/light_swing.mid
# test_patterns/complex_harmony.mid
# etc.

interpreter = FeatureInterpreter(
    encoder,
    feature_bank,
    test_patterns_dir=test_patterns_dir
)
```

**2. Manual naming**

For features that can't be auto-interpreted:

```python
# Manually examine feature and assign name
feature = bank.get_feature(feature_id=5)

# Plot activation across corpus
import matplotlib.pyplot as plt

plt.hist(feature.activation_values, bins=50)
plt.title("Feature 5 Activation Distribution")
plt.show()

# Examine songs with high activation
high_activation_indices = np.argsort(feature.activation_values)[::-1][:10]
print("Songs with highest activation:")
for idx in high_activation_indices:
    print(f"  {midi_files[idx].name}: {feature.activation_values[idx]:.3f}")

# Listen to these songs and determine what they have in common
# Then manually set name:
feature.name = "blues_feel"
feature.interpretation = "Strength of blues harmonic patterns"
feature.modality = FeatureModality.HARMONY
```

---

### Issue: Feature modality classification is wrong

**Symptom:**
A rhythm feature is classified as harmony, or vice versa.

**Solution:**

Check locality profile and manually correct:

```python
feature = bank.get_feature(name="misclassified_feature")

# Check locality profile
print(feature.locality_profile)

# If it responds strongly to rhythm transformations:
if (feature.locality_profile.rhythm_augment_sensitivity > 0.5 and
    feature.locality_profile.transpose_sensitivity < 0.2):
    # It's a rhythm feature
    feature.modality = FeatureModality.RHYTHM

# If it responds strongly to pitch transformations:
if (feature.locality_profile.transpose_sensitivity > 0.5 and
    feature.locality_profile.rhythm_augment_sensitivity < 0.2):
    # It's a harmony/melody feature
    if feature.name contains "chord":
        feature.modality = FeatureModality.HARMONY
    else:
        feature.modality = FeatureModality.MELODY
```

---

## Validation Failures

### Issue: Locality consistency check failing

**Symptom:**
```
Feature 7 failed validation:
  - locality_consistency
```

**Meaning:** Feature doesn't respond consistently to transformations.

**Diagnostic:**

```python
from midi_generator.learning.musical_locality import (
    MusicalLocalityFunctions,
    LocalityType
)

locality = MusicalLocalityFunctions()
feature = bank.get_feature(feature_id=7)

# Test consistency
test_file = Path("data/midi/test/example.mid")

# Extract feature from original
original_features = encoder.extract_semantic_features(test_file)
original_value = original_features[7]

# Transform and extract
transformed = locality.transpose(test_file, semitones=5)
transformed_features = encoder.extract_semantic_features(transformed)
transformed_value = transformed_features[7]

print(f"Original: {original_value:.3f}")
print(f"Transformed (+5 semitones): {transformed_value:.3f}")
print(f"Change: {transformed_value - original_value:.3f}")

# Do this for multiple transformations
for semitones in [2, 5, 7, 12]:
    transformed = locality.transpose(test_file, semitones=semitones)
    transformed_features = encoder.extract_semantic_features(transformed)
    change = transformed_features[7] - original_value
    print(f"+{semitones:2d} semitones: {change:+.3f}")

# Consistent feature should show similar change for same transformation
```

**Solution:**

If feature is inconsistent, it may need to be filtered out or the model retrained with higher locality weight.

---

### Issue: Musical validity check failing

**Symptom:**
```
Feature 12 failed validation:
  - musical_validity
```

**Meaning:** Feature doesn't behave in a musically meaningful way.

**Examples of invalid features:**
- Random noise
- Artifacts from overfitting
- Features that respond to file format details rather than musical content

**Diagnostic:**

```python
# Check if feature correlates with non-musical properties
feature = bank.get_feature(feature_id=12)

# Does it correlate with file size?
file_sizes = [f.stat().st_size for f in midi_files]
corr_size = np.corrcoef(feature.activation_values, file_sizes)[0, 1]
print(f"Correlation with file size: {corr_size:.3f}")

# Does it correlate with number of tracks?
import mido
num_tracks = [len(mido.MidiFile(f).tracks) for f in midi_files]
corr_tracks = np.corrcoef(feature.activation_values, num_tracks)[0, 1]
print(f"Correlation with num tracks: {corr_tracks:.3f}")

# High correlation (>0.7) suggests invalid feature
```

**Solution:**

Filter out invalid features or retrain with better regularization.

---

## Integration Issues

### Issue: Discovered parameters not showing up in registry

**Symptom:**
```python
from midi_generator.parameters.universal_registry import UniversalParameterRegistry

registry = UniversalParameterRegistry()
print(registry.list_parameters())
# Discovered parameters not in list
```

**Solution:**

Ensure parameters were registered:

```python
# In pipeline
results = pipeline.run()

# Check if registration happened
print(f"Valid features: {results['num_valid_features']}")

# Manually register if needed
for feature in results['features']:
    if feature.validation_status == "passed":
        registry.register_parameter(
            name=feature.name,
            extraction_function=feature.extraction_function,
            description=feature.interpretation,
            modality=str(feature.modality.value)
        )
```

---

### Issue: Extraction functions raising errors

**Symptom:**
```python
param_value = registry.extract_parameter("swing_ratio", Path("song.mid"))
# Raises: AttributeError, KeyError, or other error
```

**Solution:**

Wrap extraction in try-except:

```python
def safe_extract(param_name: str, midi_file: Path) -> Optional[float]:
    """Safely extract parameter, returning None on error"""
    try:
        return registry.extract_parameter(param_name, midi_file)
    except Exception as e:
        print(f"Error extracting {param_name} from {midi_file.name}: {e}")
        return None

# Use safe version
value = safe_extract("swing_ratio", Path("song.mid"))
if value is not None:
    print(f"Swing ratio: {value:.3f}")
```

---

## Data Issues

### Issue: MIDI files corrupted or unreadable

**Symptom:**
```
ValueError: Invalid MIDI file
```

**Solution:**

Filter out problematic files:

```python
import mido

def is_valid_midi(midi_file: Path) -> bool:
    """Check if MIDI file is valid"""
    try:
        midi = mido.MidiFile(midi_file)
        # Check basic properties
        if len(midi.tracks) == 0:
            return False
        if midi.length == 0:
            return False
        return True
    except:
        return False

# Filter corpus
all_files = list(corpus_dir.glob("*.mid"))
valid_files = [f for f in all_files if is_valid_midi(f)]

print(f"Valid: {len(valid_files)}/{len(all_files)}")

# Use only valid files
dataset = GapDataset(valid_files, ...)
```

---

### Issue: Corpus too small

**Symptom:**
```
Discovered features are not meaningful
Reconstruction quality is poor
```

**Minimum Requirements:**
- Training: 500+ MIDI files
- Validation: 100+ MIDI files
- Test: 100+ MIDI files

**Solutions:**

**1. Collect more data**

Public MIDI datasets:
- Lakh MIDI Dataset (176,581 files)
- MAESTRO (1,282 classical piano performances)
- MusicNet (330 classical recordings)

**2. Use data augmentation**

Automatically generate variations:

```python
from midi_generator.learning.musical_locality import (
    MusicalLocalityFunctions,
    LocalityType
)

locality = MusicalLocalityFunctions()

# For each original file, create variations
augmented_files = []
for midi_file in original_files:
    augmented_files.append(midi_file)  # Original

    # Transpose variations
    for semitones in [-3, -2, 2, 3]:
        transposed = locality.transpose(midi_file, semitones=semitones)
        augmented_files.append(transposed)

    # Velocity variations
    for factor in [0.8, 1.2]:
        scaled = locality.velocity_scale(midi_file, factor=factor)
        augmented_files.append(scaled)

# Now have 11x more files
print(f"Augmented corpus: {len(augmented_files)} files")
```

---

## Debugging Tips

### Enable verbose logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)

# Now see detailed logs from all modules
```

### Check intermediate outputs

```python
# After each phase, inspect results
results = pipeline.run()

# Check gaps
print("Gap statistics:")
print(f"  Mean: {results['gap_stats']['mean_gap']:.3f}")
print(f"  Std: {results['gap_stats']['std_gap']:.3f}")

# Check training
print("Training:")
print(f"  Best epoch: {results['best_epoch']}")
print(f"  Final loss: {results['final_loss']:.4f}")

# Check features
print("Features:")
for feat in results['features']:
    print(f"  {feat.name}: {feat.confidence:.2f} confidence")
```

### Visualize features

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot feature activations
feature = bank.get_feature(name="swing_ratio")

plt.figure(figsize=(12, 4))

# Histogram
plt.subplot(1, 3, 1)
plt.hist(feature.activation_values, bins=50)
plt.title(f"{feature.name} Distribution")
plt.xlabel("Activation")
plt.ylabel("Count")

# Scatter with other feature
other_feature = bank.get_feature(name="chord_density")
plt.subplot(1, 3, 2)
plt.scatter(feature.activation_values, other_feature.activation_values, alpha=0.5)
plt.xlabel(feature.name)
plt.ylabel(other_feature.name)
plt.title("Feature Correlation")

# Locality profile
plt.subplot(1, 3, 3)
locality_values = [
    feature.locality_profile.transpose_sensitivity,
    feature.locality_profile.rhythm_augment_sensitivity,
    feature.locality_profile.velocity_scale_sensitivity,
    # ... etc
]
locality_names = ["Transpose", "Rhythm", "Velocity", ...]
plt.bar(locality_names, locality_values)
plt.title("Locality Profile")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Profile performance

```python
import cProfile
import pstats

# Profile training
profiler = cProfile.Profile()
profiler.enable()

results = pipeline.run()

profiler.disable()

# Print stats
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

---

## Getting Help

If you encounter issues not covered here:

1. **Check GitHub Issues**: https://github.com/doseedo/Do/issues
2. **Enable debug logging** and examine output
3. **Create a minimal reproducible example**
4. **Report the issue** with:
   - Python version
   - PyTorch version
   - GPU model (if using)
   - Full error traceback
   - Minimal code to reproduce
   - Expected vs actual behavior

---

**Last Updated:** November 2025
**Version:** 1.0.0
**Agent 10:** Documentation & Examples
