# YAMNet Audio Review - Quick Start

## What is This?

YAMNet is Google's audio classification model that can identify 521 different sound classes. Use it to automatically review your vocal training dataset and find problematic audio files.

## Quick Commands

### 1. Test on Sample (5 files, ~30 seconds)
```bash
python yamnet_labeling.py --max_files 5
```

### 2. Run on Full Dataset (~10 hours for 32K files)
```bash
./run_yamnet_full.sh
```

### 3. Run in Batches (Better control)
```bash
./yamnet_batch_processor.sh
```

### 4. View Results
```bash
# Statistics
python filter_by_yamnet_labels.py --analyze_only

# Review report
less vocal_training_manifest_yamnet_labeled_review.txt
```

### 5. Filter Bad Entries
```bash
# Interactive mode (recommended)
python filter_by_yamnet_labels.py --interactive

# Or specify criteria directly
python filter_by_yamnet_labels.py \
    --exclude_warnings "music,noise,static" \
    --min_confidence 0.3
```

## What Gets Detected

✅ **Good (Keep These)**
- Singing (female, male, child)
- Speech/Voice
- Choir/Chant
- Clean vocals

⚠️ **Warnings (Review These)**
- Music/Instruments in background
- Noise/Static/Distortion
- Silence
- Environmental sounds (wind, rain, etc.)

❌ **Bad (Remove These)**
- No vocal content
- Mostly silence
- Heavy noise/distortion
- Wrong content (pure music, effects, etc.)

## Typical Workflow

```bash
# 1. Label with YAMNet
python yamnet_labeling.py \
    --input_manifest vocal_training_manifest_with_alternates.json \
    --output_manifest vocal_training_manifest_yamnet_labeled.json

# 2. Review report
cat vocal_training_manifest_yamnet_labeled_review.txt | less

# 3. Filter interactively
python filter_by_yamnet_labels.py --interactive

# 4. Use clean manifest
python trainer_performervox.py \
    --manifest_json vocal_training_manifest_filtered_clean.json
```

## Files Created

After running YAMNet:
- `vocal_training_manifest_yamnet_labeled.json` - Full manifest with labels
- `vocal_training_manifest_yamnet_labeled_review.txt` - Human-readable report
- `vocal_training_manifest_filtered_clean.json` - Filtered clean manifest

## Example Output

```
Total entries: 32,016
Successfully labeled: 31,850
Entries with warnings: 8,542 (26.7%)

Top warnings:
  - Contains music: 5,234 files (16.3%)
  - Contains guitar: 3,128 files (9.8%)
  - No clear vocal content: 1,876 files (5.9%)

Top predicted classes:
  - Singing: 18,923 files (59.1%)
  - Music: 5,234 files (16.3%)
  - Speech: 3,891 files (12.2%)
```

## Quick Filtering Options

### Conservative (Clean vocals only)
```bash
--exclude_warnings "music,instrument,noise"
--min_confidence 0.5
```
Expected: ~60-70% retention

### Moderate (Some background OK)
```bash
--exclude_warnings "noise,static,distortion"
--min_confidence 0.3
```
Expected: ~80-85% retention

### Lenient (Quality check only)
```bash
--exclude_warnings "silence,heavy noise"
--min_confidence 0.2
```
Expected: ~90-95% retention

## Tips

1. **Start Small**: Test with `--max_files 100` first
2. **Resume**: Use `--skip_existing` if interrupted
3. **Iterate**: Try different filter settings
4. **Review**: Check the TXT report for edge cases
5. **Compare**: Keep both versions of manifest

## Performance

- **Speed**: ~50 files/minute
- **GPU**: Uses ~5GB VRAM
- **Time**: 32K files = ~10 hours
- **Storage**: Labels add ~2KB per entry

## Troubleshooting

**Slow processing?**
- Run on GPU (auto-detected)
- Process in batches
- Reduce max_duration (edit script)

**Out of memory?**
- Close other GPU programs
- Reduce batch size
- Process in smaller chunks

**Resume from crash?**
- Re-run with `--skip_existing`
- Checkpoints saved every 100 files

## Next Steps

1. Run labeling (overnight)
2. Review warnings in TXT file
3. Filter based on your quality needs
4. Train with clean manifest
5. Compare results!

See `YAMNET_LABELING_GUIDE.md` for detailed documentation.
