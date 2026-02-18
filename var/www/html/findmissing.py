from pathlib import Path

# === CONFIG ===
ENCODEC_DIR = Path("/home/arlo/Data/encodec_tokens")
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths4.txt")
OUTPUT_MISSING = Path("/home/arlo/Data/missing_encodec_paths.txt")

# === 1. Read all audio paths from list
with open(AUDIO_LIST_FILE, "r") as f:
    all_audio_paths = [Path(line.strip()) for line in f if line.strip().endswith(".wav")]

# === 2. Build set of EnCodec stems
encodec_stems = set(p.stem.lower() for p in ENCODEC_DIR.rglob("*.pt"))

# === 3. Find which audio files don't have a matching token
missing = []
for audio_path in all_audio_paths:
    stem = audio_path.stem.lower()
    if stem not in encodec_stems:
        missing.append(str(audio_path))

# === 4. Log results
with open(OUTPUT_MISSING, "w") as f:
    for line in missing:
        f.write(line + "\n")

print(f"🔍 Scanned {len(all_audio_paths)} audio paths.")
print(f"❌ Missing encodec tokens: {len(missing)}")
print(f"📄 Logged to: {OUTPUT_MISSING}")
