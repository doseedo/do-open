# Full Pipeline Workflow - Vocal Training Dataset

Complete workflow from raw manifest to clean training data.

## Current Status

✅ **Completed:**
1. Created alternate takes manifest (32,016 entries)
2. YAMNet labeled 1,000 entries (test run)
3. Ollama reviewed 1,000 entries → 735 clean, 265 flagged

⏳ **Next Steps:**
1. Run YAMNet on FULL dataset (32,016 entries)
2. Flag non-vocals on full dataset
3. Train with clean manifest

---

## Step 1: YAMNet Full Dataset (~10-11 hours)

### Option A: Run Full Script
```bash
./run_full_yamnet.sh
```

### Option B: Manual Command
```bash
python yamnet_labeling.py \
    --input_manifest ./vocal_training_manifest_with_alternates.json \
    --output_manifest ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --skip_existing \
    --create_report
```

**Output:** `vocal_training_manifest_yamnet_labeled_FULL.json` (32,016 entries)

---

## Step 2: Flag Non-Vocals

### Fast Method (Instant - Recommended)
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_flagged_FULL.json \
    --output_clean ./vocal_clean_FULL.json
```

**Expected Results:**
- Process 32,016 entries instantly
- Flag ~10-20% (3,200-6,400 entries)
- Clean manifest: ~25,600-28,800 entries

### Accurate Method (Ollama - 1-2 days)
```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_ollama_flagged_FULL.json \
    --output_clean ./vocal_ollama_clean_FULL.json \
    --model llama3.2:3b
```

**Expected Results:**
- Process 32,016 entries in ~18-44 hours
- Flag ~12-18% (3,800-5,800 entries)
- Clean manifest: ~26,200-28,200 entries
- More accurate flagging

### Hybrid Approach (Best of Both)
```bash
# 1. Quick keyword pass (instant)
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./keyword_flagged.json \
    --output_clean ./keyword_clean.json

# 2. Review keyword results
python -c "
import json
flagged = json.load(open('keyword_flagged.json'))
print(f'Keyword flagged: {len(flagged)}')
"

# 3. Ollama on uncertain cases only (faster)
#    Filter keyword flagged entries with medium/low confidence
python -c "
import json
flagged = json.load(open('keyword_flagged.json'))
uncertain = [e for e in flagged if e['review']['confidence'] != 'high']
with open('uncertain_only.json', 'w') as f:
    json.dump([e['full_entry'] for e in uncertain], f, indent=2)
print(f'Uncertain cases for Ollama: {len(uncertain)}')
"

# 4. Review uncertain with Ollama (much faster)
python ollama_review_yamnet.py \
    --input ./uncertain_only.json \
    --output_flagged ./ollama_uncertain_flagged.json \
    --model llama3.2:3b
```

---

## Step 3: Train with Clean Data

```bash
python trainer_performervox.py \
    --manifest_json ./vocal_clean_FULL.json \
    --checkpoint_dir /path/to/ace_step_checkpoint \
    --batch_size 4 \
    --max_steps 200000 \
    --devices 1
```

---

## File Tracking

### Input Files
- `vocal_training_manifest_with_alternates.json` - Original with alternates (32,016)

### After YAMNet
- `vocal_training_manifest_yamnet_labeled_FULL.json` - Full labeled (32,016)
- `vocal_training_manifest_yamnet_labeled_FULL_review.txt` - Human readable

### After Flagging (choose one)

**Keyword method:**
- `vocal_flagged_FULL.json` - Flagged entries
- `vocal_clean_FULL.json` - Clean manifest for training

**Ollama method:**
- `vocal_ollama_flagged_FULL.json` - Flagged entries
- `vocal_ollama_clean_FULL.json` - Clean manifest for training

---

## Current Test Results (1,000 entries)

Based on your test run:
- **Total:** 1,000 entries
- **Keep:** 735 (73.5%)
- **Flagged:** 265 (26.5%)

### Extrapolated to Full Dataset (32,016 entries):
- **Expected Keep:** ~23,500 entries (73.5%)
- **Expected Flagged:** ~8,500 entries (26.5%)

This is actually a good filtering rate!

---

## Time Estimates

| Task | Method | Time | Output |
|------|--------|------|--------|
| YAMNet Full | GPU | 10-11 hours | 32,016 labeled |
| Flagging | Keyword | <1 sec | ~23,500 clean |
| Flagging | Ollama | 18-44 hours | ~23,500 clean |
| Training | GPU | Days-weeks | Model checkpoint |

---

## Recommended Workflow

### For Speed (1 day total):
```bash
# Day 1 morning: Start YAMNet
./run_full_yamnet.sh

# Day 1 evening: YAMNet done, run keyword flagging (instant)
python flag_non_vocals.py \
    --input vocal_training_manifest_yamnet_labeled_FULL.json

# Day 1 evening: Start training
python trainer_performervox.py \
    --manifest_json vocal_clean_FULL.json
```

### For Accuracy (2-3 days total):
```bash
# Day 1: YAMNet (10-11 hours)
./run_full_yamnet.sh

# Day 2-3: Ollama review (18-44 hours)
python ollama_review_yamnet.py \
    --input vocal_training_manifest_yamnet_labeled_FULL.json

# Day 3: Start training
python trainer_performervox.py \
    --manifest_json vocal_ollama_clean_FULL.json
```

### For Best Results (2 days):
```bash
# Day 1: YAMNet + keyword flagging
./run_full_yamnet.sh
python flag_non_vocals.py --input vocal_training_manifest_yamnet_labeled_FULL.json

# Day 2: Ollama on uncertain cases only (~5K entries = 3-7 hours)
python ollama_review_yamnet.py --input uncertain_only.json

# Day 2 evening: Start training
python trainer_performervox.py --manifest_json combined_clean.json
```

---

## Commands Summary

```bash
# 1. YAMNet on full dataset
./run_full_yamnet.sh

# 2. Flag non-vocals (choose one)
python flag_non_vocals.py --input yamnet_FULL.json          # Fast
python ollama_review_yamnet.py --input yamnet_FULL.json     # Accurate

# 3. Check results
python -c "
import json
clean = json.load(open('vocal_clean_FULL.json'))
print(f'Clean entries: {len(clean)}')
"

# 4. Train
python trainer_performervox.py --manifest_json vocal_clean_FULL.json
```

---

## Why Only 1,000 Entries?

Your current `vocal_training_manifest_yamnet_labeled.json` only has 1,000 entries because:

1. YAMNet was run with `--max_files 1000` (test mode)
2. OR the input manifest only had 1,000 entries

**To process all 32,016 entries:**
- Use `run_full_yamnet.sh`
- OR run `yamnet_labeling.py` without `--max_files` flag

The Ollama script correctly processed **all** entries in the input file (1000/1000 = 100%).

---

## Next Step

Run YAMNet on the full dataset:
```bash
./run_full_yamnet.sh
```

This will label all 32,016 entries overnight.
