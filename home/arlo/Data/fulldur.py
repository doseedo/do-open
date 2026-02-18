import os
import unicodedata
import soundfile as sf
from pathlib import Path
import csv
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# === CONFIG ===
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths4.txt")
OUTPUT_CSV = Path("/home/arlo/Data/all_audio_durations.csv")
NUM_WORKERS = 4  # Use all but one CPU

# === Helpers ===
def normalize_path(p):
    return unicodedata.normalize("NFKC", p)

def get_duration(path: Path):
    try:
        info = sf.info(path)
        duration = round(info.frames / info.samplerate, 4)
        return str(path), duration
    except Exception as e:
        return str(path), None  # Could log error info if needed

# === Main ===
if __name__ == "__main__":
    with open(AUDIO_LIST_FILE, "r") as f:
        all_paths = [Path(normalize_path(line.strip())) for line in f if line.strip().endswith(".wav")]

    print(f"🔍 Found {len(all_paths)} audio files.")
    total_duration = 0.0
    rows = []

    with Pool(NUM_WORKERS) as pool:
        for result in tqdm(pool.imap_unordered(get_duration, all_paths), total=len(all_paths)):
            file_path, duration = result
            if duration is not None:
                total_duration += duration
                rows.append((file_path, duration))

    # === Save to CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["file_path", "duration_sec"])
        writer.writerows(rows)

    print(f"✅ Logged durations to {OUTPUT_CSV}")
    print(f"🕒 Total Duration: {round(total_duration, 2)} seconds ({round(total_duration / 3600, 2)} hours)")
