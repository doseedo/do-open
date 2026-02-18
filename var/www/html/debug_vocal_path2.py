#!/usr/bin/env python3
from pathlib import Path

# Test with working vocal file
test_path = "/home/arlo/gcs-bucket/protools/2025-04-07/New/ 2024_summer in a ghost town_AWelling/2024.12.07_SIGT Lead Vox/Audio Files/Audio 1.01_02.wav"
audio_path = Path(test_path)
path_parts = audio_path.parts

print(f"Audio path: {test_path}")
print(f"Path parts: {path_parts}")

# Extract session info - need to handle nested folder structure
session_folder = "2024.12.07_SIGT Lead Vox"  # This is the actual piano roll folder name
date_folder = path_parts[path_parts.index("New") - 1]

print(f"Session folder: {session_folder}")
print(f"Date folder: {date_folder}")
print(f"Audio stem: {audio_path.stem}")

# Check paths
piano_roll_path = f"/mnt/msdd/piano_rolls/{session_folder}/{audio_path.stem}.pianoroll.npy"
print(f"Piano roll path: {piano_roll_path}")
print(f"Piano roll exists: {Path(piano_roll_path).exists()}")

# Latent path
relative_path = Path(*path_parts[path_parts.index("protools"):])
latent_path = f"/mnt/msdd/dcae_latentsnew/{relative_path.as_posix()}".replace(".wav", ".pt")
print(f"Latent path: {latent_path}")
print(f"Latent exists: {Path(latent_path).exists()}")

# Encodec path - try different variations
encodec_paths = [
    f"/mnt/msdd/encodec_tokens/ 2024_summer in a ghost town_AWelling/{audio_path.stem}.pt",
    f"/mnt/msdd/encodec_tokens/2024_summer in a ghost town_AWelling/{audio_path.stem}.pt",
    f"/mnt/msdd/encodec_tokens/{session_folder}/{audio_path.stem}.pt"
]

for i, encodec_path in enumerate(encodec_paths):
    print(f"Encodec path {i+1}: {encodec_path}")
    print(f"Encodec exists: {Path(encodec_path).exists()}")

# Check what's actually in the encodec tokens folder for this session
print(f"\nChecking encodec folder contents:")
encodec_session_folder = " 2024_summer in a ghost town_AWelling"
encodec_dir = Path(f"/mnt/msdd/encodec_tokens/{encodec_session_folder}")
if encodec_dir.exists():
    files = list(encodec_dir.glob("Audio*"))[:5]
    print(f"Found {len(files)} Audio files in {encodec_dir}")
    for f in files:
        print(f"  {f.name}")
else:
    # Try without space
    encodec_dir2 = Path(f"/mnt/msdd/encodec_tokens/2024_summer in a ghost town_AWelling")
    if encodec_dir2.exists():
        files = list(encodec_dir2.glob("Audio*"))[:5]
        print(f"Found {len(files)} Audio files in {encodec_dir2}")
        for f in files:
            print(f"  {f.name}")
    else:
        print(f"Neither encodec directory exists")