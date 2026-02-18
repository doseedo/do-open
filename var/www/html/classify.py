import os
import json
import re
import librosa
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# === CONFIG ===
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths3.txt")
PROTOOLS_ROOT = Path("/home/arlo/gcs-bucket/protools")
OUTPUT_DIR = Path("/home/arlo/Data/yamnet_classifications")
CONFIDENCE_THRESHOLD = 0.1
NUM_WORKERS = 32

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load YAMNet model and class map ===
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
with open(class_map_path) as f:
    class_names = [line.strip().split(',')[2] for line in f.readlines()[1:]]

# === Helpers ===
def classify_audio(filepath):
    try:
        waveform, sr = librosa.load(filepath, sr=16000, mono=True)
        waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
        scores, _, _ = yamnet_model(waveform)
        mean_scores = tf.reduce_mean(scores, axis=0).numpy()
        top_indices = np.argsort(mean_scores)[::-1][:5]
        return [
            {"name": class_names[i], "score": float(mean_scores[i])}
            for i in top_indices if mean_scores[i] >= CONFIDENCE_THRESHOLD
        ]
    except Exception as e:
        return {"error": str(e)}

def classify_batch(paths):
    results = {}
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_file = {executor.submit(classify_audio, str(p)): p for p in paths}
        for future in as_completed(future_to_file):
            path = future_to_file[future]
            try:
                result = future.result()
                results[str(path)] = result
                print(f"✅ Done: {path.name}")
            except Exception as e:
                print(f"❌ Error: {path.name}: {e}")
                results[str(path)] = {"error": str(e)}
    return results

# === MAIN ===
session_map = {}

# Step 1: Parse all files and group by session
with open(ALL_AUDIO_PATHS) as f:
    for line in f:
        line = line.strip()
        if not line or not line.endswith(".wav"):
            continue
        audio_file = Path(line)
        try:
            relative_path = audio_file.relative_to(PROTOOLS_ROOT)
            session_parts = relative_path.parts[:3]  # e.g., ['2025-03-28', 'New', 'MySession']
            if len(session_parts) < 3:
                continue
            session_key = "/".join(session_parts)
            session_map.setdefault(session_key, []).append(audio_file)
        except ValueError:
            continue  # Skip non-protools paths

# Step 2: Classify each session
for session_key, paths in session_map.items():
    out_dir = OUTPUT_DIR / session_key
    out_path = out_dir / "yamnet_classifications.json"

    if out_path.exists():
        print(f"⏭️ Skipping already classified session: {session_key}")
        continue

    print(f"\n🚀 Session: {session_key} — {len(paths)} files")
    classified = classify_batch(paths)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(classified, f, indent=2)
    print(f"💾 Saved: {out_path}")


    out_dir = OUTPUT_DIR / session_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "yamnet_classifications.json"

    with open(out_path, "w") as f:
        json.dump(classified, f, indent=2)
    print(f"💾 Saved: {out_path}")
