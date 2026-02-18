# Alternate Takes Implementation

## Overview

Successfully implemented alternate takes functionality for vocal training. The system now identifies and uses alternate vocal takes from the same recording session to provide voice reference embeddings during training.

## Files Modified

### 1. **find_alternate_takes.py** (NEW)
Script that analyzes the vocal training manifest to find alternate takes:
- Identifies files with numbered extensions (e.g., `izaiah.12_17.wav`, `izaiah.13_18.wav`)
- Groups files from the same session folder with the same base name
- Creates a new manifest with `alternate_takes` field for each entry

**Output**: `/home/arlo/Data/vocal_training_manifest_with_alternates.json`

### 2. **dataloadvervox.py** (MODIFIED)
Updated the dataloader to use alternate takes:
- Modified `_load_voice_reference()` to accept list of dicts instead of list of paths
- Now uses `manifest_index` for direct lookup instead of searching by path
- Loads alternate take latents and extracts voice embeddings from voiced segments
- Returns `[8, 16]` reference embedding tensor

**Key changes**:
```python
# Before: alternate_takes was List[str]
# After: alternate_takes is List[Dict[str, Any]] with keys:
#   - manifest_index: int
#   - audio_path: str
#   - number_extension: str
```

### 3. **trainer_performervox.py** (MODIFIED)
Updated default manifest path:
```python
# Changed from:
ap.add_argument("--manifest_json", type=str, default="./final_training_manifest_final.json")

# To:
ap.add_argument("--manifest_json", type=str, default="./vocal_training_manifest_with_alternates.json")
```

## Manifest Statistics

- **Total entries**: 32,016
- **Entries with alternates**: 16,241 (50.7%)
- **Total alternate references**: 479,662
- **Average alternates per entry**: 29.5

## How It Works

### 1. Alternate Take Detection
Files are considered alternate takes if they:
- Are in the same session folder
- Have the same base filename
- Have different numbered extensions (format: `basename.XX_YY.wav`)

**Example**:
```
/protools/2025-06-26/New/Welcome to your life/Audio Files/
  ├── izaiah.12_17.wav  ← Main take
  ├── izaiah.13_18.wav  ← Alternate take
  ├── izaiah.14_20.wav  ← Alternate take
  └── ...
```

### 2. Voice Reference Loading

During training, the dataloader:
1. Checks if entry has `alternate_takes` field
2. Applies dropout (`voice_reference_dropout=0.20`, so 80% use rate)
3. Randomly selects one alternate take
4. Loads the alternate's latent and amplitude conditioning
5. Extracts only voiced segments (amp > threshold)
6. Pools voiced latents to create `[8, 16]` voice embedding

### 3. Integration with Training

The reference latent is returned in the batch:
```python
batch = {
    "latents": ...,              # [B, 8, 16, T]
    "encodec_tokens": ...,       # [B, 8, T_fast]
    "conds": {...},
    "reference_latent": ...,     # [B, 8, 16] or None
    "vocal_conditioning": ...,
    ...
}
```

The trainer can use this reference embedding to:
- Condition the model on singer identity
- Improve voice consistency across generations
- Enable voice transfer capabilities

## Usage

### Generate Manifest with Alternates
```bash
python find_alternate_takes.py
```

### Train with Alternates
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_with_alternates.json \
    --voice_reference_dropout 0.20
```

### Adjust Alternate Usage
```python
dataset = PerformerAIVocalDataset(
    json_path="vocal_training_manifest_with_alternates.json",
    voice_reference_dropout=0.10,  # Use alternates 90% of the time
    voice_amp_threshold=0.06,       # Threshold for voiced segment detection
    ...
)
```

## Verification

Run verification script to check manifest structure:
```bash
python verify_alternates_manifest.py
```

Expected output:
- ✅ All structure checks passed
- ✅ Valid manifest_index references
- ✅ No self-references
- ✅ All required fields present

## Notes

- Alternates are only used for voice reference, not as separate training samples
- The current entry is automatically excluded from alternate selection
- If no alternates exist, `reference_latent` will be `None` in the batch
- Voice reference dropout helps prevent overfitting to specific singer characteristics
