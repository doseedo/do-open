import os
import unicodedata
import time
import gc
from pathlib import Path
from multiprocessing import Process, Lock
import numpy as np
import librosa
import soundfile as sf

# === CONFIG ===
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths4.txt")
OUTPUT_DIR = Path("/mnt/msdd/audio_features4")
DONE_LOG = Path("/home/arlo/Data/chroma_done2.txt")
PROTOOLS_ROOTS = [
    Path("/home/arlo/gcs-bucket/protools"),
    Path("/home/arlo/gcs-bucket/protoolsA")
]
SKIP_COUNT = 0
SKIP_DONE = True
NUM_WORKERS = 25

CHROMA_HOP_LENGTH = 640
CHROMA_N_FFT = 2048
SAMPLE_RATE = 24000

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

done_files = set()
if SKIP_DONE and DONE_LOG.exists():
    with open(DONE_LOG, "r") as f:
        done_files = set(line.strip() for line in f if line.strip())

def normalize_path(p):
    return unicodedata.normalize("NFKC", p)

def load_paths():
    with open(ALL_AUDIO_PATHS, "r") as f:
        all_paths = [normalize_path(line.strip()) for line in f if line.strip().endswith(".wav")]
    return all_paths[SKIP_COUNT:]

def resolve_relative_to_roots(audio_file):
    for root in PROTOOLS_ROOTS:
        try:
            return audio_file.relative_to(root)
        except ValueError:
            continue
    return None

def process_worker(paths, lock, worker_id):
    total = len(paths)
    print(f"🎹 Worker {worker_id}: {total} files")

    new_done = []

    for idx, raw_path in enumerate(paths, 1):
        audio_file = Path(raw_path)
        try:
            real_path = str(audio_file.resolve())
        except FileNotFoundError:
            print(f"🚫 [W{worker_id}] Unresolved: {audio_file}")
            continue

        if SKIP_DONE and real_path in done_files:
            print(f"⏭️ [W{worker_id}] Skipping {idx}/{total}: {audio_file.name}")
            continue

        relative_path = resolve_relative_to_roots(audio_file)
        if relative_path is None:
            print(f"⚠️ [W{worker_id}] Not under any protools root: {audio_file.name}")
            continue

        try:
            if "New" in relative_path.parts:
                session_folder = relative_path.parts[relative_path.parts.index("New") + 1]
            elif "Prev" in relative_path.parts:
                session_folder = relative_path.parts[relative_path.parts.index("Prev") + 1]
            else:
                print(f"⚠️ [W{worker_id}] No session folder found in path: {relative_path}")
                continue
            relative_session = Path(session_folder)
        except (ValueError, IndexError):
            print(f"⚠️ [W{worker_id}] Could not extract session: {relative_path}")
            continue

        out_path = OUTPUT_DIR / relative_session / (audio_file.stem + ".chroma.npy")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"⏩ [W{worker_id}] Already exists: {out_path.name}")
            new_done.append(real_path)
            continue

        try:
            y, sr = sf.read(audio_file)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != SAMPLE_RATE:
                y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
                sr = SAMPLE_RATE

            chroma = librosa.feature.chroma_stft(
                y=y,
                sr=sr,
                hop_length=CHROMA_HOP_LENGTH,
                n_fft=CHROMA_N_FFT
            )

            np.save(out_path, chroma)
            print(f"✅ [W{worker_id}] {idx}/{total} Saved: {out_path.name}")
            new_done.append(real_path)
        except Exception as e:
            print(f"❌ [W{worker_id}] {idx}/{total} {audio_file.name}: {e}")
        finally:
            gc.collect()

    with lock:
        with open(DONE_LOG, "a") as f:
            for path in new_done:
                f.write(path + "\n")

# === MAIN ===
if __name__ == "__main__":
    all_paths = load_paths()
    print(f"🎵 Total chroma files to process: {len(all_paths)}")

    per_worker = len(all_paths) // NUM_WORKERS
    chunks = [all_paths[i * per_worker:(i + 1) * per_worker] for i in range(NUM_WORKERS)]
    chunks[-1].extend(all_paths[NUM_WORKERS * per_worker:])

    lock = Lock()
    workers = []
    for i, chunk in enumerate(chunks):
        p = Process(target=process_worker, args=(chunk, lock, i))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print("🏁 All chroma extraction done.")
