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

# === DRUM KEYWORDS ===
EXCLUDE_KEYWORDS = [
    "Kick", "KICK", "KickIn", "KickOut", "KickSub", "Kik", "Kck", "KIK", "kick", "kik",
    "Snare", "Snr", "SNR", "Sn", "SN", "SnrTop", "SnrBtm", "SNARE", "snare", "sn",
    "HiHat", "HH", "ClosedHat", "OpenHat", "Hat", "HAT", "hihat",
    "Tom", "RackTom", "FloorTom", "RTom", "FTom", "TOM",
    "Cymbal", "Cym", "Crash", "Ride", "Splash", "China", "Stack", "CYM", "CYMBAL",
    "OH", "Overhead", "OHL", "OHR", "OVERHEAD",
    "Perc", "Tamb", "Cowbell", "Clap", "Shaker", "Triangle", "PERC", "CLAP",
    "Drum", "DRUM", "drum"
]

def is_drum_file(path: Path) -> bool:
    """Check if filename or parent folders contain drum-related keywords."""
    name = path.name
    parts = list(path.parts)
    combined = " ".join([name] + parts)
    return any(kw in combined for kw in EXCLUDE_KEYWORDS)

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

def select_model(sr, model_24, model_48):
    if sr == 48000:
        return model_48, 48000
    elif sr == 24000:
        return model_24, 24000
    else:
        chosen = model_48 if abs(sr - 48000) < abs(sr - 24000) else model_24
        target_sr = 48000 if chosen == model_48 else 24000
        print(f"⚠️ Resampling {sr}Hz → {target_sr}Hz")
        return chosen, target_sr


def process_worker(paths, gpu_id, lock):
    device = torch.device(f"cuda:{gpu_id}")
    print(f"🚀 Worker using {device} — {len(paths)} files")

    model_24 = EncodecModel.encodec_model_24khz()
    model_24.set_target_bandwidth(6.0)
    model_24.to(device).eval()

    model_48 = EncodecModel.encodec_model_48khz()
    model_48.set_target_bandwidth(6.0)
    model_48.to(device).eval()

    new_done = []
    stereo_log_path = OUTPUT_DIR / "stereo_paths.txt"

    for idx, raw_path in enumerate(paths, 1):
        audio_file = Path(raw_path)
        try:
            real_path = str(audio_file.resolve())
        except FileNotFoundError:
            print(f"🚫 Path does not resolve: {audio_file}")
            continue

        # === Skip drum files ===
        if is_drum_file(audio_file):
            print(f"🥁 [{gpu_id}] Skipping drum file: {audio_file.name}")
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
            waveform, sr = torchaudio.load(audio_file)  # shape: (channels, samples)

            # === If stereo, log it, then convert to mono instead of skipping ===
            if waveform.shape[0] == 2:
                print(f"🎙️ [{gpu_id}] Found stereo; converting to mono: {audio_file.name}")
                with lock:
                    with open(stereo_log_path, "a") as f:
                        f.write(real_path + "\n")

            # === Force 24kHz / mono for Encodec-24k ===
            model, target_sr = model_24, 24000
            if sr != target_sr or waveform.shape[0] != 1:
                waveform = convert_audio(waveform, sr, target_sr, 1)  # downmix to mono

            with torch.no_grad():
                tokens = model.encode(waveform.unsqueeze(0).to(device))

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

    per_worker = len(all_paths) // len(GPU_IDS)
    chunks = [all_paths[i * per_worker:(i + 1) * per_worker] for i in range(len(GPU_IDS))]
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
