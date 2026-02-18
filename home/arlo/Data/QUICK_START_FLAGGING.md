# Quick Start: Flag Non-Vocal Entries

After YAMNet labeling completes, flag problematic entries.

## Fast Method (Recommended First)

### 1. Run Keyword-Based Flagging (~instant)

```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output_flagged ./vocal_flagged.json \
    --output_clean ./vocal_clean.json
```

**Output:**
- `vocal_flagged.json` - Problematic entries
- `vocal_clean.json` - Clean manifest for training

### 2. Review Results

```bash
# Show statistics
python -c "
import json
flagged = json.load(open('vocal_flagged.json'))
clean = json.load(open('vocal_clean.json'))
total = len(flagged) + len(clean)
print(f'Total: {total}')
print(f'Clean: {len(clean)} ({100*len(clean)/total:.1f}%)')
print(f'Flagged: {len(flagged)} ({100*len(flagged)/total:.1f}%)')
"

# View flagged reasons
python -c "
import json
from collections import Counter
flagged = json.load(open('vocal_flagged.json'))
reasons = [item['review']['reason'].split(';')[0] for item in flagged]
for reason, count in Counter(reasons).most_common(10):
    print(f'{count:5d}  {reason}')
"
```

### 3. Use Clean Manifest

```bash
python trainer_performervox.py \
    --manifest_json ./vocal_clean.json \
    --checkpoint_dir /path/to/checkpoint
```

---

## Ollama Method (More Accurate, Slower)

### Setup (One Time)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull model
ollama pull llama3.2:3b
```

### Run Review

```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output_flagged ./vocal_ollama_flagged.json \
    --output_clean ./vocal_ollama_clean.json \
    --model llama3.2:3b
```

**Time:** ~2-5 sec/entry
- 32K entries ≈ 18-44 hours
- Saves checkpoints every 50 entries
- Can resume with `--skip_existing`

---

## What Gets Flagged?

### ❌ Flagged (Removed):
- Silence (>70% confidence)
- Pure instrumental/music
- Heavy noise/distortion
- No vocal content detected
- Environmental sounds only

### ✅ Kept:
- Singing (any type)
- Speech/voice/vocals
- Music WITH vocals
- Choir/chant

---

## Quick Commands

```bash
# Fast flagging (instant)
python flag_non_vocals.py --input yamnet_labeled.json

# Strict mode (flag more)
python flag_non_vocals.py --input yamnet_labeled.json --strict

# Ollama review (accurate)
python ollama_review_yamnet.py --input yamnet_labeled.json --model llama3.2:3b

# Test Ollama on 100 entries first
python ollama_review_yamnet.py --input yamnet_labeled.json --max_entries 100

# Compare both methods
python flag_non_vocals.py --input yamnet_labeled.json --output_flagged keyword.json
python ollama_review_yamnet.py --input yamnet_labeled.json --output_flagged ollama.json --max_entries 1000
```

---

## Files Created

After running:

1. **`vocal_flagged.json`** - Problematic entries with reasons
2. **`vocal_clean.json`** - Clean manifest (use for training)
3. **Stats in terminal** - Summary of what was flagged

---

## Workflow

```bash
# 1. YAMNet labeling (already done)
✅ vocal_training_manifest_yamnet_labeled.json

# 2. Flag non-vocals (choose one)
python flag_non_vocals.py --input yamnet_labeled.json    # Fast
# OR
python ollama_review_yamnet.py --input yamnet_labeled.json  # Accurate

# 3. Review
cat vocal_flagged.json | jq '.[0:5]' | less

# 4. Train with clean data
python trainer_performervox.py --manifest_json vocal_clean.json
```

---

## Pro Tips

### Combine Both Methods

```bash
# 1. Quick pass with keywords
python flag_non_vocals.py --input yamnet.json --output_clean keyword_clean.json

# 2. Then Ollama on uncertain cases
python ollama_review_yamnet.py --input keyword_clean.json --max_entries 5000

# Uses keyword for obvious cases, Ollama for edge cases
```

### Check Before/After

```bash
# Before
cat vocal_training_manifest_yamnet_labeled.json | jq '. | length'

# After
cat vocal_clean.json | jq '. | length'

# Difference = flagged entries
```

### Review Specific Index

```bash
# View entry details
python -c "
import json
data = json.load(open('vocal_flagged.json'))
entry = data[0]
print(f\"Index: {entry['original_index']}\")
print(f\"Path: {entry['audio_path']}\")
print(f\"Reason: {entry['review']['reason']}\")
print(f\"Top class: {entry['yamnet_labels']['top_class']}\")
"
```

---

See `FLAGGING_NON_VOCALS.md` for detailed documentation.
