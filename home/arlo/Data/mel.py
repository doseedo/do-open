import os
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# === CONFIG ===
AUDIO_PATHS_FILE = Path("/home/arlo/Data/all_audio_paths4.txt")
INPUT_ROOT = Path("/home/arlo/gcs-bucket/protools")
OUTPUT_ROOT = Path("/home/arlo/gcs-bucket/mel_spectrograms")
SAMPLE_RATE = 48000
N_MELS = 192
HOP_LENGTH = 256
N_FFT = 2048
NUM_WORKERS = 20
SKIP_COUNT = 121000  # ✅ set this to skip N files from top of list

INPUT_ROOTS = [
    Path("/home/arlo/gcs-bucket/protools"),
    Path("/home/arlo/gcs-bucket/protoolsA")
]

def get_relative_path(path):
    for root in INPUT_ROOTS:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    raise ValueError(f"{path} is not in any known input roots: {INPUT_ROOTS}")


# === Load audio paths from file ===
with open(AUDIO_PATHS_FILE, "r") as f:
    all_paths = [Path(line.strip()) for line in f if line.strip().endswith(".wav")]

audio_files = all_paths[SKIP_COUNT:]
print(f"🎵 Loaded {len(audio_files)} audio paths (after skipping {SKIP_COUNT})")

# === Worker function ===
def process_file(audio_path: Path):
    try:
        rel_path = get_relative_path(audio_path)

        mel_file = OUTPUT_ROOT / rel_path.with_suffix(".npy")
        if mel_file.exists():
            return f"⏭️ Skipped: {mel_file.name}"

        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS,
                                             hop_length=HOP_LENGTH, n_fft=N_FFT)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        mel_file.parent.mkdir(parents=True, exist_ok=True)
        np.save(mel_file, mel_db)
        return f"✅ Saved: {mel_file.name}"
    except Exception as e:
        return f"❌ Failed: {audio_path.name} — {e}"

# === Parallel processing ===
if __name__ == "__main__":
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap_unordered(process_file, audio_files), total=len(audio_files)))

    print("🏁 Done. Summary:")
    for r in results:
        if r.startswith("❌"):
            print(r)
