# Pipeline Status & Next Steps

## Current Status ✅

### Completed:
1. ✅ **Alternate Takes Manifest** - 32,016 entries
2. ✅ **YAMNet Test Run** - 1,000 entries (100% success)
3. ✅ **Ollama Test Review** - 1,000 entries reviewed
   - Keep: 735 (73.5%)
   - Flagged: 265 (26.5%)

### Test Results:
Based on 1,000 entry test:
- **Success rate:** 100% YAMNet labeling
- **Warning rate:** 36.8% have warnings
- **Keep rate:** 73.5% suitable for training
- **Projected clean dataset:** ~23,500 entries (from 32,016)

---

## Why Ollama Only Processed 1,000 Entries

**Answer:** The input file `vocal_training_manifest_yamnet_labeled.json` only has 1,000 entries.

This happened because:
- YAMNet was run with `--max_files 1000` (test mode)
- Ollama correctly processed **all** entries in that file (1000/1000)

**Solution:** Run YAMNet on the FULL manifest (32,016 entries)

---

## Next Steps

### STEP 1: YAMNet Full Dataset (~10-11 hours)

**Run this command:**
```bash
./run_full_yamnet.sh
```

**Or manually:**
```bash
python yamnet_labeling.py \
    --input_manifest ./vocal_training_manifest_with_alternates.json \
    --output_manifest ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --skip_existing \
    --create_report
```

**What it does:**
- Labels all 32,016 entries
- Takes ~10-11 hours on GPU
- Saves checkpoints every 100 files
- Can resume if interrupted

**Output:**
- `vocal_training_manifest_yamnet_labeled_FULL.json`
- `vocal_training_manifest_yamnet_labeled_FULL_review.txt`

---

### STEP 2: Flag Non-Vocals (After YAMNet Completes)

**Option A: Fast (Instant - Recommended)**
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_flagged_FULL.json \
    --output_clean ./vocal_clean_FULL.json
```

**Expected output:**
- Clean: ~23,500 entries (73.5%)
- Flagged: ~8,500 entries (26.5%)
- Time: <1 second

**Option B: Accurate (1-2 days)**
```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_ollama_flagged_FULL.json \
    --output_clean ./vocal_ollama_clean_FULL.json \
    --model llama3.2:3b
```

**Expected output:**
- Clean: ~23,500-24,500 entries
- Flagged: ~7,500-8,500 entries
- Time: 18-44 hours
- More accurate flagging

---

### STEP 3: Train (After Flagging)

```bash
python trainer_performervox.py \
    --manifest_json ./vocal_clean_FULL.json \
    --checkpoint_dir /path/to/ace_step_checkpoint \
    --batch_size 4 \
    --max_steps 200000 \
    --devices 1
```

---

## Quick Commands

### Check Current Status
```bash
python check_pipeline_status.py
```

### Run Full YAMNet (Start Now!)
```bash
./run_full_yamnet.sh
```

### After YAMNet, Flag Non-Vocals
```bash
# Fast method
python flag_non_vocals.py --input vocal_training_manifest_yamnet_labeled_FULL.json

# OR accurate method
python ollama_review_yamnet.py --input vocal_training_manifest_yamnet_labeled_FULL.json
```

### Then Train
```bash
python trainer_performervox.py --manifest_json vocal_clean_FULL.json
```

---

## Timeline

### Fast Track (1-2 days):
- **Day 1 (10-11 hours):** YAMNet full dataset
- **Day 1 (<1 sec):** Keyword flagging
- **Day 1-∞:** Training starts

### Accurate Track (2-3 days):
- **Day 1 (10-11 hours):** YAMNet full dataset
- **Day 2-3 (18-44 hours):** Ollama review
- **Day 3-∞:** Training starts

### Hybrid Track (1.5-2 days):
- **Day 1 (10-11 hours):** YAMNet full dataset
- **Day 1 (<1 sec):** Keyword flagging
- **Day 2 (3-7 hours):** Ollama on uncertain cases only
- **Day 2-∞:** Training starts

---

## File Organization

```
Current Files:
├── vocal_training_manifest_with_alternates.json (32,016) ✅
├── vocal_training_manifest_yamnet_labeled.json (1,000) ✅ TEST
├── vocal_training_manifest_ollama_flagged.json (265) ✅ TEST
└── vocal_training_manifest_ollama_clean.json (735) ✅ TEST

After YAMNet Full:
├── vocal_training_manifest_yamnet_labeled_FULL.json (32,016) ⏳
└── vocal_training_manifest_yamnet_labeled_FULL_review.txt ⏳

After Flagging:
├── vocal_flagged_FULL.json (~8,500) ⏳
└── vocal_clean_FULL.json (~23,500) ⏳ ← USE FOR TRAINING
```

---

## Important Notes

1. **Ollama worked correctly** - it processed all 1,000 entries in the input file
2. **You need to run YAMNet on the FULL dataset** to get all 32,016 entries labeled
3. **Start YAMNet now** - it will take 10-11 hours
4. **While YAMNet runs**, you can prepare by ensuring Ollama is set up (if using)

---

## Commands to Run Right Now

```bash
# 1. Check you have the alternates manifest
ls -lh vocal_training_manifest_with_alternates.json

# 2. Start YAMNet on full dataset (do this now!)
./run_full_yamnet.sh

# 3. Come back in 10-11 hours, then run flagging
# (Wait for YAMNet to complete first)
```

---

## Expected Final Results

Based on test data (1,000 entries):
- **Starting:** 32,016 entries
- **After YAMNet:** 32,016 labeled (100%)
- **After flagging:** ~23,500 clean entries (73.5%)
- **Training dataset:** ~23,500 high-quality vocal tracks

This is a good filtering rate - you'll have a clean dataset with the problematic entries removed!
