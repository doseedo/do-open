from pathlib import Path

# === CONFIG ===
CHROMA_DIR = Path("/mnt/msdd/audio_features")
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths4.txt")
OUTPUT_MISSING = Path("/home/arlo/Data/missing_chroma_paths.txt")

# === 1. Read all audio paths from list
with open(AUDIO_LIST_FILE, "r") as f:
    all_audio_paths = [Path(line.strip()) for line in f if line.strip().endswith(".wav")]

# === 2. Build set of chroma stems (remove .chroma)
chroma_stems = set()
for f in CHROMA_DIR.rglob("*.chroma.npy"):
    stem = f.stem.lower()
    if stem.endswith(".chroma"):
        stem = stem.replace(".chroma", "")
    chroma_stems.add(stem)

# === 3. Compare against audio stems
missing = []
for audio_path in all_audio_paths:
    stem = audio_path.stem.lower()
    if stem not in chroma_stems:
        missing.append(str(audio_path))

# === 4. Log results
with open(OUTPUT_MISSING, "w") as f:
    for line in missing:
        f.write(line + "\n")

print(f"🔍 Scanned {len(all_audio_paths)} audio paths.")
print(f"❌ Missing chroma files: {len(missing)}")
print(f"📄 Logged to: {OUTPUT_MISSING}")
