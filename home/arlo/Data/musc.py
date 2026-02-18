import os
import unicodedata
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from multiprocessing import Process, Queue
from tqdm import tqdm

# === CONFIG ===
ALL_AUDIO_PATHS = Path("/home/arlo/Data/audio_files_needing_conditioning.txt")
OUTPUT_DIR = Path("/mnt/msdd/newerconditioning")
DONE_LOG = Path("/home/arlo/Data/newconditioning_done.txt")
PROTOOLS_ROOTS = [
    Path("/home/arlo/gcs-bucket/protools"),
    Path("/home/arlo/gcs-bucket/protoolsA")
]
SKIP_COUNT = 0

SKIP_DONE = False

# --- NEW: OPTION TO SKIP DRUMS ---
SKIP_DRUMS = True

# --- PROCESSING PARAMS ---
SAMPLE_RATE = 44100
HOP_LENGTH = 4096
N_FFT = 8192
FILES_PER_WORKER = 32
NUM_WORKERS = 28

# --- DURATION LIMITS IN SECONDS ---
MIN_DURATION = 4.0
MAX_DURATION = 12 * 60.0 # 12 minutes

# Keywords for filtering files (unchanged)
exclude_keywords = [
    "Kick", "KICK", "KickIn", "KickOut", "KickSub", "Kik", "Kck", "KIK", "kick", "kik",
    "Snare", "Snr", "SNR", "Sn", "SN", "SnrTop", "SnrBtm", "SNARE", "snare", "sn",
    "HiHat", "HH", "ClosedHat", "OpenHat", "Hat", "HAT", "hihat",
    "Tom", "RackTom", "FloorTom", "RTom", "FTom", "TOM",
    "Cymbal", "Cym", "Crash", "Ride", "Splash", "China", "Stack", "CYM", "CYMBAL",
    "OH", "Overhead", "OHL", "OHR", "OVERHEAD",
    "Perc", "Tamb", "Cowbell", "Clap", "Shaker", "Triangle", "PERC", "CLAP",
    "Drum", "Drums", "Drumkit", "Kit", "KIT", "DRUM", "DRUMKIT"
]

def normalize_path(p):
    return unicodedata.normalize("NFKC", p)

def is_drum_file(name):
    lname = name.lower()
    return any(kw.lower() in lname for kw in exclude_keywords)

def load_paths():
    with open(ALL_AUDIO_PATHS, "r") as f:
        return [normalize_path(line.strip())
               for line in f if line.strip().endswith(".wav")][SKIP_COUNT:]

def resolve_relative_to_roots(audio_file):
    for root in PROTOOLS_ROOTS:
        try:
            return audio_file.relative_to(root)
        except ValueError:
            continue
    return None

def worker(worker_id, file_queue, result_queue):
    """
    Worker process to handle feature extraction for a batch of files.
    """
    while True:
        file_batch = file_queue.get()
        if file_batch is None:
            break

        processed = []
        for audio_file in file_batch:
            try:
                audio_file = Path(audio_file)
                real_path = str(audio_file.resolve())

                relative_path = resolve_relative_to_roots(audio_file)
                if not relative_path:
                    continue

                session_folder = None
                if "New" in relative_path.parts:
                    session_folder = relative_path.parts[relative_path.parts.index("New") + 1]
                elif "Prev" in relative_path.parts:
                    session_folder = relative_path.parts[relative_path.parts.index("Prev") + 1]
                else:
                    session_folder = relative_path.parts[relative_path.parts.index("2025-05-11") + 1]

                out_base = OUTPUT_DIR / session_folder / audio_file.stem
                out_base.parent.mkdir(parents=True, exist_ok=True)

                onset_path = out_base.with_suffix(".onsets.npy")
                rframe_path = out_base.with_suffix(".rframe.npy")
                rbend_path = out_base.with_suffix(".rbend.npy")
                amp_path = out_base.with_suffix(".amp.npy")
                f0_path = out_base.with_suffix(".f0.npy")
                f0_masked_path = out_base.with_suffix(".f0_masked.npy")

                if rframe_path.exists() and rbend_path.exists() and amp_path.exists():
                    processed.append(real_path)
                    continue

                y, sr = sf.read(audio_file)

                duration = len(y) / sr
                if not (MIN_DURATION <= duration <= MAX_DURATION):
                    tqdm.write(f"⏭️ [Worker {worker_id}] Skipping due to duration ({duration:.2f}s): {audio_file.name}")
                    continue

                if y.ndim > 1:
                    y = np.mean(y, axis=1)

                y_resampled = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

                amp_env = librosa.feature.rms(
                    y=y_resampled, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
                amp_env = amp_env / np.max(amp_env) if np.max(amp_env) > 0 else amp_env

                onset_frames = librosa.onset.onset_detect(
                    y=y_resampled, sr=SAMPLE_RATE, hop_length=HOP_LENGTH, backtrack=False)
                onsets_binary = np.zeros(len(amp_env), dtype=np.float32)
                valid_onsets = onset_frames[onset_frames < len(onsets_binary)]
                onsets_binary[valid_onsets] = 1.0

                # This logic remains, but is now redundant if SKIP_DRUMS is True.
                # It serves as a fallback if you set SKIP_DRUMS to False.
                if is_drum_file(audio_file.name):
                    tqdm.write(f"⏩ [Worker {worker_id}] Skipping pitch for drum file: {audio_file.name}")
                    n_frames = len(amp_env)
                    f0 = np.zeros(n_frames, dtype=np.float32)
                    voiced_flag = np.zeros(n_frames, dtype=np.float32)
                else:
                    f0, voiced_flag, _ = librosa.pyin(
                        y=y_resampled,
                        fmin=librosa.note_to_hz('C2'),
                        fmax=librosa.note_to_hz('C7'),
                        sr=SAMPLE_RATE,
                        frame_length=N_FFT,
                        hop_length=HOP_LENGTH
                    )
                    f0[~voiced_flag] = 0

                rframe = voiced_flag.astype(np.float32)

                with np.errstate(divide='ignore', invalid='ignore'):
                    rbend = 12 * np.log2(f0 / 440.0)
                    rbend[~np.isfinite(rbend)] = 0
                rbend *= rframe

                assert len(amp_env) == len(onsets_binary) == len(f0) == len(rframe) == len(rbend)

                np.save(amp_path, amp_env.astype(np.float32))
                np.save(onset_path, onsets_binary.astype(np.float32))
                np.save(rframe_path, rframe.astype(np.float32))
                np.save(rbend_path, rbend.astype(np.float32))

                if not is_drum_file(audio_file.name):
                    f0_masked = f0 * rframe
                    np.save(f0_path, f0.astype(np.float32))
                    np.save(f0_masked_path, f0_masked.astype(np.float32))

                processed.append(real_path)
                tqdm.write(f"✅ [Worker {worker_id}] Processed: {audio_file.name}")

            except Exception as e:
                tqdm.write(f"❌ [Worker {worker_id}] FAILED on {audio_file.name}: {str(e)}")

        result_queue.put(processed)

if __name__ == "__main__":
    all_paths = load_paths()
    print(f"🎵 Total files loaded: {len(all_paths)}")

    # --- NEW: FILTER DRUMS BEFORE PROCESSING ---
    if SKIP_DRUMS:
        print("🥁 Skipping drum files...")
        paths_to_process = [p for p in all_paths if not is_drum_file(Path(p).name)]
        print(f"🎧 Found {len(paths_to_process)} non-drum files to process.")
    else:
        paths_to_process = all_paths
    # --- END NEW ---

    if SKIP_DONE:
        done_files = set()
        if DONE_LOG.exists():
            with open(DONE_LOG, "r") as f:
                done_files = set(line.strip() for line in f if line.strip())
            print(f"Found {len(done_files)} previously completed files.")

        if done_files:
            print("🔎 Filtering out completed files (this may take a moment)...")
            # This loop is now only run when necessary
            paths_to_process_filtered = []
            for p in tqdm(paths_to_process, desc="Filtering completed files"):
                if str(Path(p).resolve()) not in done_files:
                    paths_to_process_filtered.append(p)
            paths_to_process = paths_to_process_filtered

    print(f"🚀 Remaining files to process: {len(paths_to_process)}")

    file_queue = Queue()
    result_queue = Queue()

    workers = []
    for i in range(NUM_WORKERS):
        p = Process(target=worker, args=(i, file_queue, result_queue))
        p.start()
        workers.append(p)

    print("📦 Populating work queue...")
    current_batch = []
    for path in tqdm(paths_to_process, desc="Queueing files"):
        current_batch.append(path)
        if len(current_batch) >= FILES_PER_WORKER:
            file_queue.put(current_batch)
            current_batch = []
    if current_batch:
        file_queue.put(current_batch)

    for _ in range(NUM_WORKERS):
        file_queue.put(None)

    processed_count = 0
    with tqdm(total=len(paths_to_process), desc="Overall Progress") as pbar:
        num_batches = (len(paths_to_process) + FILES_PER_WORKER - 1) // FILES_PER_WORKER
        for _ in range(num_batches):
            try:
                results = result_queue.get(timeout=60)
                if results:
                    batch_size = len(results)
                    processed_count += batch_size
                    with open(DONE_LOG, "a") as f:
                        f.write("\n".join(results) + "\n")
                    pbar.update(batch_size)
            except Exception:
                tqdm.write("⌛ Waiting for results...")
                if not any(p.is_alive() for p in workers):
                    break

    for p in workers:
        p.join()

    print(f"\n🏁 Processed {processed_count} new files using {NUM_WORKERS} workers.")