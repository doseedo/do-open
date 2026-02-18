import os
from pathlib import Path

# === CONFIG ===
MOUNT_PATH = Path("/home/arlo/gcs-bucket")
OUTPUT_FILE = Path("/home/arlo/Data/all_audio_paths3.txt")
EXCLUDED_DIRS = ["Prev", "Bounced Files"]
INCLUDE_EXTS = [".wav", ".aif", ".aiff"]

def is_valid_audio_file(path: Path):
    if any(excl in path.parts for excl in EXCLUDED_DIRS):
        return False
    return path.suffix.lower() in INCLUDE_EXTS

def build_path_list():
    print(f"🔍 Scanning: {MOUNT_PATH}")
    count = 0
    with open(OUTPUT_FILE, "w") as out:
        for path in MOUNT_PATH.rglob("*"):
            if path.is_file() and is_valid_audio_file(path):
                out.write(str(path) + "\n")
                count += 1
    print(f"✅ Saved {count} audio paths to {OUTPUT_FILE}")

if __name__ == "__main__":
    build_path_list()
