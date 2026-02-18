import os
import unicodedata
import soundfile as sf
from pathlib import Path
import csv

# === CONFIG ===
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths4.txt")
OUTPUT_CSV = Path("/home/arlo/Data/trombone_durations.csv")

# Violin-related keywords
VIOLIN_KEYWORDS = [
    "Trombone", "TROMBONE", "TB", "tb", "trombone", "Tbn", "TBN", "tbn", "Bone", "BONE"
]

# === Helper ===
def normalize_path(p):
    return unicodedata.normalize("NFKC", p)

def is_violin(name):
    lname = name.lower()
    return any(k.lower() in lname for k in VIOLIN_KEYWORDS)

# === Main ===
with open(AUDIO_LIST_FILE, "r") as f:
    audio_paths = [Path(normalize_path(line.strip())) for line in f if line.strip().endswith(".wav")]

violin_files = [p for p in audio_paths if is_violin(p.name)]

print(f"🎻 Found {len(violin_files)} trombone-related files.")

total_duration = 0.0

with open(OUTPUT_CSV, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["file_path", "duration_sec"])

    for path in violin_files:
        try:
            info = sf.info(path)
            duration = round(info.frames / info.samplerate, 4)
            total_duration += duration
            writer.writerow([str(path), duration])
        except Exception as e:
            print(f"❌ Error loading {path.name}: {e}")

print(f"✅ Logged durations to {OUTPUT_CSV}")
print(f"🕒 Total Duration: {round(total_duration, 2)} seconds ({round(total_duration / 3600, 2)} hours)")
