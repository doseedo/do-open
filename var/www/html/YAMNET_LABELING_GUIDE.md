# YAMNet Audio Labeling Guide

## Overview

This guide explains how to use YAMNet (Google's audio classification model) to label and review your vocal training dataset. YAMNet can identify 521 different audio classes and help you find problematic entries in your manifest.

## Files

### 1. `yamnet_labeling.py`
Main script that runs YAMNet on your audio files and creates a labeled manifest.

### 2. `filter_by_yamnet_labels.py`
Analyze and filter the labeled manifest to remove bad entries.

## Quick Start

### Step 1: Run YAMNet Labeling

Process your entire manifest:
```bash
python yamnet_labeling.py \
    --input_manifest ./vocal_training_manifest_with_alternates.json \
    --output_manifest ./vocal_training_manifest_yamnet_labeled.json \
    --create_report
```

This will:
- Load YAMNet model (521 audio classes)
- Process each audio file (analyzes 30 seconds from middle)
- Identify top 10 predictions with scores
- Detect warnings/issues
- Save labeled manifest
- Create human-readable review report

**Options:**
- `--max_files N`: Only process N files (for testing)
- `--skip_existing`: Resume from previous run (skips already labeled)
- `--create_report`: Generate TXT review file (default: True)

### Step 2: Review Results

Check the review report:
```bash
cat vocal_training_manifest_yamnet_labeled_review.txt
```

Or analyze statistics:
```bash
python filter_by_yamnet_labels.py --analyze_only \
    --input ./vocal_training_manifest_yamnet_labeled.json
```

### Step 3: Filter Bad Entries

Remove entries with issues:
```bash
python filter_by_yamnet_labels.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output ./vocal_training_manifest_filtered_clean.json \
    --exclude_warnings "music,guitar,piano,noise" \
    --min_confidence 0.3 \
    --require_vocal
```

Or use interactive mode:
```bash
python filter_by_yamnet_labels.py --interactive
```

## YAMNet Output Structure

Each entry gets a `yamnet_labels` field:

```json
{
  "audio_path": "/path/to/audio.wav",
  "yamnet_labels": {
    "status": "success",
    "top_predictions": [
      {
        "class": "Singing",
        "score": 0.85,
        "percentage": 85.0
      },
      ...
    ],
    "all_predictions": [...],  // Top 10
    "top_class": "Singing",
    "top_score": 0.85,
    "confidence_stats": {
      "mean": 0.82,
      "min": 0.65
    },
    "warnings": [
      "Contains music",
      "Contains guitar"
    ],
    "num_frames": 62
  }
}
```

## Warning Types

YAMNet automatically detects these issues:

### 1. **Non-Vocal Content**
- Music, Musical instruments
- Guitar, Piano, Drums, etc.
- Electronic music, Synthesizer

**Example**: Track labeled "Music" instead of "Singing"

### 2. **Noise/Interference**
- Noise, Static, Hum
- White noise, Pink noise
- Crackle, Click, Pop, Distortion

**Example**: Recording has background static

### 3. **Missing Vocal Content**
- No singing/speech detected in top 5 predictions
- Silence is top prediction

**Example**: File is mostly silence

### 4. **Environmental Sounds**
- Wind, Rain, Thunder, Water
- Birds, Dogs, Traffic
- Doors, Footsteps

**Example**: Outdoor recording with wind noise

## Common Filtering Strategies

### Conservative (Keep Only Clean Vocals)
```bash
python filter_by_yamnet_labels.py \
    --exclude_warnings "music,instrument,guitar,piano,drum,noise,static" \
    --min_confidence 0.5 \
    --require_vocal
```

### Moderate (Allow Some Music)
```bash
python filter_by_yamnet_labels.py \
    --exclude_warnings "noise,static,distortion" \
    --min_confidence 0.3 \
    --require_vocal
```

### Quality Check Only (Remove Clear Problems)
```bash
python filter_by_yamnet_labels.py \
    --exclude_warnings "silence,noise,static,distortion" \
    --min_confidence 0.2 \
    --require_vocal
```

## Analyzing Results

### View Statistics
```bash
python filter_by_yamnet_labels.py --analyze_only --input manifest.json
```

Shows:
- Total entries processed
- Most common warnings
- Top predicted classes
- Confidence distribution

### Example Output
```
Total entries: 32,016
Successfully labeled: 31,850
Failed: 166

Entries with warnings: 8,542 (26.7%)

Top warnings:
  - Contains music: 5,234 files (16.3%)
  - Contains guitar: 3,128 files (9.8%)
  - No clear vocal content detected: 1,876 files (5.9%)
  - Contains noise: 445 files (1.4%)

Top predicted classes:
  - Singing: 18,923 files (59.1%)
  - Music: 5,234 files (16.3%)
  - Speech: 3,891 files (12.2%)
  - Female singing: 2,667 files (8.3%)

Confidence statistics:
  - Mean: 0.742
  - Median: 0.831
  - Min: 0.023
  - Max: 0.998
```

## Tips

### 1. Start with Small Sample
Test on a few files first:
```bash
python yamnet_labeling.py --max_files 100
```

### 2. Resume if Interrupted
The script saves checkpoints every 100 files:
```bash
python yamnet_labeling.py --skip_existing
```

### 3. Check Warnings First
Review the most problematic entries:
```bash
grep -A 5 "Warnings:" vocal_training_manifest_yamnet_labeled_review.txt | head -50
```

### 4. Iterate Filtering
Start conservative, then relax:
```bash
# Round 1: Very strict
python filter_by_yamnet_labels.py --min_confidence 0.7 --output clean_v1.json

# Round 2: Moderate
python filter_by_yamnet_labels.py --min_confidence 0.4 --output clean_v2.json

# Round 3: Lenient
python filter_by_yamnet_labels.py --min_confidence 0.2 --output clean_v3.json
```

## YAMNet Vocal Classes

YAMNet recognizes these vocal-related classes:
- Singing
- Female singing
- Male singing
- Child singing
- Choir
- Chant
- Yodeling
- Speech
- Male speech
- Female speech
- Child speech
- Narration
- Conversation
- Vocal music

## Performance

**Processing Speed:**
- ~1-2 seconds per file
- ~50 files/minute
- 32,000 files ≈ 10-11 hours

**Memory Usage:**
- YAMNet model: ~5GB GPU
- Audio loading: ~500MB RAM per file

**Recommendations:**
- Run overnight for large datasets
- Use `--skip_existing` to resume
- Process in batches with `--max_files`

## Troubleshooting

### Model Download Issues
```bash
# Manually download YAMNet
export TFHUB_CACHE_DIR=/home/arlo/.cache/tfhub
python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"
```

### Out of Memory
- Reduce max_duration in script (default: 30s)
- Process fewer files at once
- Close other GPU programs

### Audio Loading Errors
- Check file permissions
- Verify audio format (WAV/MP3/FLAC supported)
- Check for corrupted files

## Integration with Training

After filtering, use the clean manifest:
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_filtered_clean.json
```

Or compare before/after:
```bash
# Count entries
echo "Original: $(cat vocal_training_manifest_with_alternates.json | jq '. | length')"
echo "After YAMNet filter: $(cat vocal_training_manifest_filtered_clean.json | jq '. | length')"
```

## Next Steps

1. ✅ Run YAMNet labeling
2. ✅ Review warnings in report
3. ✅ Filter based on your quality threshold
4. ✅ Analyze filtered statistics
5. ✅ Use clean manifest for training
6. 🔄 Iterate if needed

For questions or issues, check the review report first to understand what YAMNet detected.
