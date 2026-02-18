# Flagging Non-Vocal Entries - Guide

After YAMNet labeling, use these tools to identify and remove problematic entries.

## Two Approaches

### 1. **Ollama LLM Review** (Intelligent, Slower)
Uses AI to understand context and make nuanced decisions.

### 2. **Keyword-Based** (Fast, Rule-Based)
Simple pattern matching, very fast.

---

## Option 1: Ollama LLM Review

### Setup

1. **Install Ollama** (if not already installed):
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

2. **Pull a model**:
```bash
# Fast, lightweight (recommended)
ollama pull llama3.2:3b

# Or larger for better accuracy
ollama pull llama3.2:7b
```

### Usage

**Full dataset:**
```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output_flagged ./vocal_training_manifest_ollama_flagged.json \
    --output_clean ./vocal_training_manifest_ollama_clean.json \
    --model llama3.2:3b
```

**Test on sample:**
```bash
python ollama_review_yamnet.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --max_entries 100 \
    --model llama3.2:3b
```

### Features

✅ Understands context (e.g., "music with vocals" → KEEP)
✅ Handles ambiguous cases better
✅ Can distinguish background music from pure instrumental
✅ Resume capability (saves checkpoints every 50 entries)

### Performance

- **Speed:** ~2-5 seconds per entry (depends on model)
- **32K entries:** ~18-44 hours
- **Accuracy:** High (understands nuance)

---

## Option 2: Keyword-Based (Fast)

### Usage

**Lenient mode (recommended):**
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output_flagged ./vocal_training_manifest_flagged.json \
    --output_clean ./vocal_training_manifest_clean.json
```

**Strict mode (more aggressive):**
```bash
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --strict
```

### Features

✅ Very fast (~instant for 32K entries)
✅ No dependencies
✅ Deterministic results
✅ Good for obvious cases

### Performance

- **Speed:** ~0.1 seconds total for 32K entries
- **Accuracy:** Good for clear cases, may miss nuanced situations

---

## Decision Criteria

### Both methods FLAG:

❌ **Silence** (>70% confidence)
❌ **Pure instrumental music** (>80% music, no vocals in top 3)
❌ **Heavy noise/distortion**
❌ **No vocal content** in top predictions
❌ **Environmental sounds only**

### Both methods KEEP:

✅ **Singing** (any type)
✅ **Speech/voice**
✅ **Choir/chant**
✅ **Music with vocals** (vocals in top 3)

---

## Which to Use?

### Use **Ollama** if:
- You want highest accuracy
- You have time (~1-2 days for full dataset)
- You have Ollama installed
- Dataset has ambiguous cases

### Use **Keyword-Based** if:
- You want immediate results
- You don't have Ollama
- Dataset is straightforward
- You'll manually review later anyway

### Use **Both**:
1. Run keyword-based first (instant)
2. Review flagged entries
3. Use Ollama on uncertain cases only

---

## Output Files

### Flagged Manifest
Contains only problematic entries:
```json
[
  {
    "original_index": 123,
    "audio_path": "/path/to/file.wav",
    "yamnet_labels": {...},
    "review": {
      "decision": "FLAG",
      "reason": "Pure instrumental music, no vocals detected",
      "confidence": "high"
    },
    "full_entry": {...}
  }
]
```

### Clean Manifest
Original manifest with flagged entries removed. Use this for training:
```bash
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_clean.json
```

---

## Example Workflow

```bash
# 1. YAMNet labeling (done)
# Output: vocal_training_manifest_yamnet_labeled.json

# 2. Quick keyword-based flagging
python flag_non_vocals.py \
    --input ./vocal_training_manifest_yamnet_labeled.json \
    --output_flagged ./keyword_flagged.json \
    --output_clean ./keyword_clean.json

# 3. Review statistics
python -c "
import json
flagged = json.load(open('keyword_flagged.json'))
print(f'Flagged: {len(flagged)}')
for item in flagged[:5]:
    print(f\"  - {item['review']['reason']}\")
"

# 4. If you want higher accuracy, use Ollama on flagged entries
python ollama_review_yamnet.py \
    --input ./keyword_flagged.json \
    --model llama3.2:3b

# 5. Use clean manifest for training
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_clean.json
```

---

## Comparison

| Feature | Ollama | Keyword |
|---------|--------|---------|
| **Speed** | 2-5 sec/entry | Instant |
| **Accuracy** | High | Good |
| **Nuance** | ✅ Understands context | ❌ Rule-based |
| **Dependencies** | Ollama + model | None |
| **Resume** | ✅ Checkpoints | N/A |
| **32K dataset** | ~1-2 days | <1 second |

---

## Tips

### 1. **Start with Keyword**
```bash
# Fast first pass
python flag_non_vocals.py --input manifest.json
# Review results
cat vocal_training_manifest_flagged.json | jq '.[].review.reason' | sort | uniq -c
```

### 2. **Use Ollama for Edge Cases**
```bash
# Review uncertain entries with Ollama
python ollama_review_yamnet.py --max_entries 1000
```

### 3. **Compare Both**
```bash
# Run both, compare results
python flag_non_vocals.py --input manifest.json --output_flagged keyword_flagged.json
python ollama_review_yamnet.py --input manifest.json --output_flagged ollama_flagged.json

# Check differences
python -c "
import json
k = set(i['original_index'] for i in json.load(open('keyword_flagged.json')))
o = set(i['original_index'] for i in json.load(open('ollama_flagged.json')))
print(f'Keyword only: {len(k - o)}')
print(f'Ollama only: {len(o - k)}')
print(f'Both: {len(k & o)}')
"
```

### 4. **Adjust Strictness**
```bash
# More aggressive
python flag_non_vocals.py --strict

# More lenient (default)
python flag_non_vocals.py
```

---

## Expected Results

For a typical vocal dataset:

- **Keyword flagging:** 10-20% flagged
  - Silence: ~5%
  - Pure music: ~8%
  - Noise: ~2%
  - Other: ~5%

- **Ollama flagging:** 12-18% flagged
  - More accurate in edge cases
  - Better at detecting subtle issues

---

## Next Steps

After flagging:

1. ✅ Review flagged entries (optional)
2. ✅ Use clean manifest for training
3. ✅ Compare training results

```bash
# Train with cleaned data
python trainer_performervox.py \
    --manifest_json ./vocal_training_manifest_clean.json \
    --batch_size 4 \
    --max_steps 200000
```
