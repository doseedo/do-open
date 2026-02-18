import json
from pathlib import Path

INDEX_PATH = Path("/home/arlo/Data/train_index.jsonl")

total_sec = 0.0
count = 0

with open(INDEX_PATH) as f:
    for line in f:
        try:
            entry = json.loads(line)
            duration = float(entry.get("duration_sec", 0))
            total_sec += duration
            count += 1
        except Exception as e:
            print(f"⚠️ Skipped malformed line: {e}")

hours = total_sec / 3600
print(f"✅ Total entries: {count}")
print(f"⏱️  Total duration: {total_sec:.2f} sec ({hours:.2f} hours)")
