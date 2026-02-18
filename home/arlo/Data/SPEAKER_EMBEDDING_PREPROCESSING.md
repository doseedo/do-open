# Speaker Embedding Preprocessing Guide

## Overview

Batch preprocess speaker embeddings from alternate takes to speed up training. Instead of extracting embeddings on-the-fly during training (90ms per file), we extract once and save to disk for instant loading.

---

## Performance Comparison

| Method | Time per Sample | Training Impact |
|--------|-----------------|-----------------|
| **On-the-fly (CPU)** | ~90ms | Bottleneck in dataloader |
| **On-the-fly (GPU)** | ~50ms | Still slow + GPU context issues |
| **Precomputed (disk)** | <1ms | No bottleneck ✅ |

### Training Speedup
- **Without preprocessing:** ~2-3 sec/batch (bottlenecked by embedding extraction)
- **With preprocessing:** ~0.5 sec/batch (full GPU utilization)

---

## Step 1: Preprocess Speaker Embeddings

### Basic Usage (GPU)
```bash
python preprocess_speaker_embeddings.py \
    --manifest ./vocal_training_manifest_yamnet_filtered.json \
    --output_dir /mnt/msdd/speaker_embeddings \
    --output_manifest ./vocal_training_manifest_with_speaker_embs.json \
    --batch_size 100 \
    --device cuda
```

### Options

```bash
--manifest              # Input manifest path
--output_dir            # Where to save embeddings (default: /mnt/msdd/speaker_embeddings)
--output_manifest       # Output manifest with embedding paths
--batch_size            # Files per batch (GPU stays warm, default: 100)
--device               # cuda or cpu
--num_workers          # Parallel GPU workers (if multiple GPUs)
--skip_existing        # Skip files that already have embeddings
```

### CPU Usage (Slower)
```bash
python preprocess_speaker_embeddings.py \
    --manifest ./vocal_training_manifest_yamnet_filtered.json \
    --device cpu
```

---

## Step 2: What Gets Created

### Output Structure
```
/mnt/msdd/speaker_embeddings/
├── a1b2c3d4_LDVOX_WET_01_spk.pt          # [256] speaker embedding
├── e5f6g7h8_Vocals_01_spk.pt
├── ...
└── [hash]_[filename]_spk.pt
```

### Updated Manifest
The output manifest will have new fields:

```json
{
  "audio_path": "/path/to/file.wav",
  "speaker_embedding_path": "/mnt/msdd/speaker_embeddings/hash_file_spk.pt",  // NEW
  "alternate_takes": [
    {
      "audio_path": "/path/to/alternate.wav",
      "speaker_embedding_path": "/mnt/msdd/speaker_embeddings/hash_alt_spk.pt",  // NEW
      "manifest_index": 123
    }
  ],
  ...
}
```

---

## Step 3: Use Fast Dataloader

### Update trainer_performervox.py

Replace import:
```python
# OLD
from dataloadvervox import PerformerAIVocalDataset, collate_latent_cond_vocal

# NEW
from dataloadvervox_fast import PerformerAIVocalDatasetFast, collate_latent_cond_vocal
```

Update dataloader initialization:
```python
ds = PerformerAIVocalDatasetFast(  # Changed class name
    json_path=self.manifest_json,
    conditioning_dropout={"piano_roll":0.15, "amp":0.10, "rbend":0.10, "rframe":0.05},
    use_trim=True,
    window_slow=getattr(self.hparams, "window_slow", 256),
    vocal_conditioning_dropout=0.05,
    voice_reference_dropout=0.20,
    use_precomputed_embeddings=True,  # NEW: Use precomputed embeddings
)
```

---

## Step 4: Run Preprocessing

### Expected Output

```
================================================================================
Speaker Embedding Preprocessing
================================================================================
Manifest: ./vocal_training_manifest_yamnet_filtered.json
Output dir: /mnt/msdd/speaker_embeddings
Device: cuda
Batch size: 100

Loading manifest...
Loaded 28208 entries

Collecting audio files...
Found 35421 unique audio files  # Main files + alternates

Processing 355 batches...

🚀 Using GPU: NVIDIA A100-SXM4-40GB

Batch 1/355 (100 files)
  ✅ Success: 98/100
Batch 2/355 (100 files)
  ✅ Success: 100/100
...

================================================================================
SUMMARY
================================================================================
Total processed: 35421
Success: 35215
Failed: 206

Missing files by type:
  - File not found: 206

Updating manifest with embedding paths...
✅ Updated manifest saved to: ./vocal_training_manifest_with_speaker_embs.json

✅ Done! Embeddings saved to: /mnt/msdd/speaker_embeddings
✅ Updated manifest: ./vocal_training_manifest_with_speaker_embs.json
```

### Timing Estimates

**GPU (A100):**
- 35,000 files × 50ms = **~30 minutes**

**GPU (V100/T4):**
- 35,000 files × 90ms = **~50 minutes**

**CPU:**
- 35,000 files × 90ms = **~50 minutes** (but no GPU memory)

---

## Step 5: Resume Interrupted Preprocessing

If preprocessing gets interrupted, resume with:

```bash
python preprocess_speaker_embeddings.py \
    --manifest ./vocal_training_manifest_yamnet_filtered.json \
    --skip_existing  # Skip already processed files
```

This will check existing embeddings and only process missing ones.

---

## Step 6: Verify Embeddings

### Quick Check
```python
import torch
from pathlib import Path

emb_dir = Path("/mnt/msdd/speaker_embeddings")
embeddings = list(emb_dir.glob("*_spk.pt"))

print(f"Total embeddings: {len(embeddings)}")

# Check a sample
sample = torch.load(embeddings[0])
print(f"Shape: {sample.shape}")  # Should be [256]
print(f"Dtype: {sample.dtype}")  # Should be torch.float32
```

---

## Step 7: Training with Precomputed Embeddings

### Before (On-the-fly extraction)
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_yamnet_filtered.json \
    --batch_size 4
# Dataloader: ~2-3 sec/batch (bottlenecked)
```

### After (Precomputed embeddings)
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_with_speaker_embs.json \
    --batch_size 4
# Dataloader: ~0.5 sec/batch (fast) ✅
```

---

## How It Works

### Preprocessing (`preprocess_speaker_embeddings.py`)

1. **Load manifest** and collect all unique audio files (main + alternates)
2. **For each audio file:**
   - Load and resample to 16kHz
   - Extract speaker embedding via Resemblyzer: `encoder.embed_utterance(audio)` → `[256]`
   - Save to disk: `/mnt/msdd/speaker_embeddings/{hash}_{filename}_spk.pt`
3. **Update manifest** with `speaker_embedding_path` fields
4. **Save updated manifest**

### Fast Loading (`dataloadvervox_fast.py`)

1. **Check if precomputed embedding exists:**
   ```python
   emb_path = alternate_take.get("speaker_embedding_path")
   if emb_path and Path(emb_path).exists():
       speaker_emb = torch.load(emb_path, map_location="cpu")  # <1ms
       return speaker_emb
   ```

2. **Fallback to on-the-fly extraction** (if precomputed missing):
   ```python
   # Load audio, resample, extract embedding (slow)
   audio = torchaudio.load(audio_path)
   speaker_emb = self._cpu_encoder.embed_utterance(audio)
   return speaker_emb
   ```

---

## Benefits

### ✅ **4-6x Faster Training**
- Eliminates dataloader bottleneck
- Full GPU utilization during training

### ✅ **No CUDA Context Issues**
- Resemblyzer runs once during preprocessing
- No GPU conflicts in dataloader workers

### ✅ **Disk Storage Efficient**
- 256 floats × 4 bytes = 1 KB per embedding
- 35,000 embeddings = ~35 MB total

### ✅ **Resumable**
- `--skip_existing` flag skips already processed files
- Can interrupt and resume anytime

---

## Troubleshooting

### Issue: "CUDA out of memory" during preprocessing
**Solution:** Reduce batch size or use CPU
```bash
--batch_size 50  # Smaller batches
# OR
--device cpu     # Use CPU (slower but no memory issues)
```

### Issue: "No such file or directory" for audio
**Solution:** Check audio paths in manifest are correct
```bash
python -c "
import json
manifest = json.load(open('manifest.json'))
for entry in manifest[:10]:
    print(entry['audio_path'])
"
```

### Issue: Slow preprocessing on CPU
**Solution:** Use GPU or run in background
```bash
nohup python preprocess_speaker_embeddings.py --device cuda > preprocess.log 2>&1 &
tail -f preprocess.log  # Monitor progress
```

### Issue: Embeddings not loading during training
**Solution:** Check manifest has `speaker_embedding_path` fields
```bash
python -c "
import json
manifest = json.load(open('manifest_with_embs.json'))
print('Has speaker_embedding_path:', 'speaker_embedding_path' in manifest[0])
"
```

---

## File Sizes

### Embeddings Only
- **1 file:** ~1 KB
- **35,000 files:** ~35 MB
- **100,000 files:** ~100 MB

### Training Data (for reference)
- **DCAE latents:** ~200 GB
- **EnCodec tokens:** ~50 GB
- **Piano rolls:** ~20 GB
- **Speaker embeddings:** ~35 MB ✅ (negligible)

---

## Alternative: On-the-Fly with CPU Encoder

If you don't want to preprocess, the original `dataloadvervox.py` uses on-the-fly extraction with a CPU-based Resemblyzer encoder to avoid CUDA issues:

```python
# In dataloadvervox.py
if not hasattr(self, 'speaker_encoder'):
    from resemblyzer import VoiceEncoder
    self.speaker_encoder = VoiceEncoder()  # CPU by default
    self.speaker_encoder.eval()
```

**Trade-off:**
- ✅ No preprocessing needed
- ❌ Slower training (2-3 sec/batch vs 0.5 sec/batch)

---

## Recommendation

**For serious training (200k+ steps):**
→ Use preprocessing (`preprocess_speaker_embeddings.py` + `dataloadvervox_fast.py`)

**For quick experiments (<10k steps):**
→ Use on-the-fly (`dataloadvervox.py`)

---

## Quick Start Commands

```bash
# 1. Preprocess embeddings (one-time, ~30 min on GPU)
python preprocess_speaker_embeddings.py \
    --manifest ./vocal_training_manifest_yamnet_filtered.json \
    --output_manifest ./vocal_training_manifest_with_speaker_embs.json \
    --device cuda

# 2. Train with precomputed embeddings (fast)
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_with_speaker_embs.json \
    --batch_size 4 \
    --max_steps 200000
```

Done! 🚀
