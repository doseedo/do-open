import os
import torch
import unicodedata
import time
import gc
from pathlib import Path
from encodec import EncodecModel
from encodec.utils import convert_audio
import torchaudio
from multiprocessing import Process, Queue, Lock

# === CONFIG ===
ALL_AUDIO_PATHS = Path("/home/arlo/Data/my_audio_paths.txt")
OUTPUT_DIR = Path("/mnt/msdd/encodec_tokens")
DONE_LOG = Path("/home/arlo/Data/encodec_done.txt")
PROTOOLS_ROOTS = [
    Path("/home/arlo/gcs-bucket/protools"),
    Path("/home/arlo/gcs-bucket/protoolsA")
]
SKIP_COUNT = 0
SKIP_DONE = False
GPU_IDS = [0]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# === Load previously completed paths ===
done_files = set()
if SKIP_DONE and DONE_LOG.exists():
    with open(DONE_LOG, "r") as f:
        done_files = set(line.strip() for line in f if line.strip())

# === Helpers ===
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

def process_worker(paths, gpu_id, lock):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"🚀 Worker using {device} — {len(paths)} files")

    # We’ll always encode at 24 kHz mono
    model_24 = EncodecModel.encodec_model_24khz()
    model_24.set_target_bandwidth(6.0)
    model_24.to(device).eval()

    new_done = []
    stereo_log_path = OUTPUT_DIR / "downmixed_paths.txt"

    for idx, raw_path in enumerate(paths, 1):
        audio_file = Path(raw_path)
        try:
            real_path = str(audio_file.resolve())
        except FileNotFoundError:
            print(f"🚫 Path does not resolve: {audio_file}")
            continue

        if SKIP_DONE and real_path in done_files:
            print(f"⏭️ [{gpu_id}] Skipping {idx}: {audio_file.name}")
            continue
        if not audio_file.exists():
            print(f"🚫 [{gpu_id}] Missing {idx}: {audio_file.name}")
            continue

        relative_path = resolve_relative_to_roots(audio_file)
        if relative_path is None:
            print(f"⚠️ [{gpu_id}] Skipping unrelated {idx}: {audio_file.name}")
            continue

        try:
            if "New" in relative_path.parts:
                session_folder = relative_path.parts[relative_path.parts.index("New") + 1]
            elif "Prev" in relative_path.parts:
                session_folder = relative_path.parts[relative_path.parts.index("Prev") + 1]
            else:
                session_folder = relative_path.parts[relative_path.parts.index("2025-05-11") + 1]

            relative_session = Path(session_folder)
        except Exception as e:
            print(f"⚠️ [{gpu_id}] Failed to extract session: {e}")
            continue

        out_path = OUTPUT_DIR / relative_session / (audio_file.stem + ".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists():
            print(f"⏩ [{gpu_id}] Already exists: {out_path.name}")
            new_done.append(real_path)
            continue

        try:
            waveform, sr = torchaudio.load(audio_file)  # (channels, samples)
            channels = waveform.shape[0]

            # 👉 Always process as 24 kHz mono
            # If not 24kHz or not mono, convert (downmix multi-channel to mono)
            if sr != 24000 or channels != 1:
                # Log files that needed downmix/resample (includes stereo & other multi‑channel)
                if channels != 1:
                    with lock:
                        with open(stereo_log_path, "a") as f:
                            f.write(real_path + "\n")
                    print(f"🎙️ [{gpu_id}] Downmixing {channels}ch → mono & resampling → 24kHz: {audio_file.name}")
                waveform = convert_audio(waveform, sr, 24000, 1)  # (1, n)

            # Encode with Encodec 24 kHz (expects batch, time)
            with torch.no_grad():
                tokens = model_24.encode(waveform.unsqueeze(0).to(device))  # (1, 1, n) → tokens structure per Encodec

            torch.save(tokens, out_path)
            print(f"✅ [{gpu_id}] {idx} Saved: {out_path.name}")
            new_done.append(real_path)

        except Exception as e:
            print(f"❌ [{gpu_id}] {idx} {audio_file.name}: {e}")
        finally:
            if 'tokens' in locals():
                del tokens
            torch.cuda.empty_cache()
            gc.collect()

    with lock:
        with open(DONE_LOG, "a") as f:
            for path in new_done:
                f.write(path + "\n")


# === MAIN ===
if __name__ == "__main__":
    all_paths = load_paths()
    print(f"🎧 Loaded {len(all_paths)} audio files (after skipping {SKIP_COUNT})")

    per_worker = len(all_paths) // len(GPU_IDS) if GPU_IDS else 0
    chunks = [all_paths[i * per_worker:(i + 1) * per_worker] for i in range(len(GPU_IDS))]
    if GPU_IDS:
        chunks[-1].extend(all_paths[len(GPU_IDS) * per_worker:])

    lock = Lock()
    workers = []
    for gpu_id, chunk in zip(GPU_IDS, chunks):
        p = Process(target=process_worker, args=(chunk, gpu_id, lock))
        p.start()
        workers.append(p)

    for p in workers:
        p.join()

    print("🏁 All workers done.")
