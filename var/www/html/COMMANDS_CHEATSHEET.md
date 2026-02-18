# Commands Cheatsheet - Vocal Training Pipeline

Quick reference for all pipeline commands.

## Check Status

```bash
# Check current pipeline status
python check_pipeline_status.py

# Check manifest sizes
ls -lh vocal_*.json | awk '{print $5, $9}'
```

---

## YAMNet Labeling

### Full Dataset (32,016 entries)
```bash
# Easy way
./run_full_yamnet.sh

# Manual way
python yamnet_labeling.py \
    --input_manifest ./vocal_training_manifest_with_alternates.json \
    --output_manifest ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --skip_existing \
    --create_report
```

### Resume if Interrupted
```bash
# YAMNet saves checkpoints every 100 entries
# Just re-run the same command
./run_full_yamnet.sh
```

---

## Flag Non-Vocals

### Fast Method (Instant)
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_flagged_FULL.json \
    --output_clean ./vocal_clean_FULL.json
```

### Strict Mode (More Aggressive)
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --strict
```

### Ollama Method (Accurate)
```bash
# One-time setup
curl -fsSL https://ollama.ai/install.sh | sh
ollama pull llama3.2:3b

# Run review
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_ollama_flagged_FULL.json \
    --output_clean ./vocal_ollama_clean_FULL.json \
    --model llama3.2:3b
```

### Test Ollama First
```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --max_entries 100 \
    --model llama3.2:3b
```

---

## Review Results

### Show Statistics
```bash
# Count entries
python -c "
import json
clean = json.load(open('vocal_clean_FULL.json'))
flagged = json.load(open('vocal_flagged_FULL.json'))
total = len(clean) + len(flagged)
print(f'Total: {total:,}')
print(f'Clean: {len(clean):,} ({100*len(clean)/total:.1f}%)')
print(f'Flagged: {len(flagged):,} ({100*len(flagged)/total:.1f}%)')
"
```

### Show Flagging Reasons
```bash
python -c "
import json
from collections import Counter
flagged = json.load(open('vocal_flagged_FULL.json'))
reasons = [e['review']['reason'].split(';')[0] for e in flagged]
for r, c in Counter(reasons).most_common(10):
    print(f'{c:5d}  {r}')
"
```

### View Specific Entry
```bash
# View entry details
python -c "
import json
data = json.load(open('vocal_flagged_FULL.json'))
entry = data[0]
print('Index:', entry['original_index'])
print('Path:', entry['audio_path'])
print('Reason:', entry['review']['reason'])
print('Top:', entry['yamnet_labels']['top_class'])
"
```

---

## Training

### Basic Training
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_clean_FULL.json \
    --checkpoint_dir /path/to/ace_step_checkpoint \
    --batch_size 4 \
    --max_steps 200000
```

### With Alternate Takes
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_clean_FULL.json \
    --checkpoint_dir /path/to/checkpoint \
    --batch_size 4 \
    --voice_reference_dropout 0.20 \
    --voice_amp_threshold 0.06
```

---

## File Sizes

```bash
# Check all manifest sizes
du -h vocal_*.json | sort -h

# Count entries in each
for f in vocal_*.json; do
    echo "$f: $(python3 -c "import json; print(len(json.load(open('$f'))))" 2>/dev/null || echo 'error')"
done
```

---

## Useful One-Liners

### Compare Manifests
```bash
# Show difference between keyword and Ollama flagging
python -c "
import json
k = set(e['original_index'] for e in json.load(open('vocal_flagged_FULL.json')))
o = set(e['original_index'] for e in json.load(open('vocal_ollama_flagged_FULL.json')))
print(f'Keyword only: {len(k - o)}')
print(f'Ollama only: {len(o - k)}')
print(f'Both flagged: {len(k & o)}')
"
```

### Extract Indices
```bash
# Get all flagged indices
python -c "
import json
flagged = json.load(open('vocal_flagged_FULL.json'))
indices = [e['original_index'] for e in flagged]
print(','.join(map(str, indices[:10])))  # First 10
"
```

### Create Subset Manifest
```bash
# Create manifest with first N entries
python -c "
import json
data = json.load(open('vocal_clean_FULL.json'))
subset = data[:1000]
with open('vocal_clean_1k.json', 'w') as f:
    json.dump(subset, f, indent=2)
print(f'Created subset: {len(subset)} entries')
"
```

---

## Troubleshooting

### YAMNet Issues
```bash
# Check CUDA
nvidia-smi

# Test on single file
python yamnet_labeling.py --max_files 1

# Check model
python -c "import tensorflow_hub as hub; hub.load('https://tfhub.dev/google/yamnet/1')"
```

### Ollama Issues
```bash
# Check Ollama
ollama --version
ollama list

# Test model
ollama run llama3.2:3b "test"

# Pull model again
ollama pull llama3.2:3b
```

### Import Issues
```bash
# Test imports
python -c "from trainer_performervox import Pipeline; print('✅')"
python -c "from dataloadvervox import PerformerAIVocalDataset; print('✅')"
python -c "from conditioning_encodervox_simple import PerformanceConditionEncoderVocalSimple; print('✅')"
```

---

## Full Pipeline (Copy-Paste)

```bash
# 1. Check status
python check_pipeline_status.py

# 2. Run YAMNet (10-11 hours)
./run_full_yamnet.sh

# 3. Flag non-vocals (instant)
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled_FULL.json \
    --output_flagged ./vocal_flagged_FULL.json \
    --output_clean ./vocal_clean_FULL.json

# 4. Review
python -c "
import json
clean = json.load(open('vocal_clean_FULL.json'))
print(f'Clean entries: {len(clean):,}')
"

# 5. Train
python trainer_performervox.py \
    --manifest_json ./vocal_clean_FULL.json \
    --checkpoint_dir /path/to/checkpoint \
    --batch_size 4
```

---

## Quick Reference

| Task | Command | Time |
|------|---------|------|
| Check status | `python check_pipeline_status.py` | <1s |
| YAMNet full | `./run_full_yamnet.sh` | 10-11h |
| Flag (fast) | `python flag_non_vocals.py --input ...` | <1s |
| Flag (Ollama) | `python ollama_review_yamnet.py --input ...` | 18-44h |
| Train | `python trainer_performervox.py --manifest_json ...` | days |
