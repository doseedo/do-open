#!/usr/bin/env python3
from pathlib import Path

# Test with one known vocal file
test_path = "/home/arlo/gcs-bucket/protools/2025-03-28/New/29Sep_Jocelyn_Vox_Sess_DONE/Audio Files/Audio 1.01_02.wav"
audio_path = Path(test_path)
path_parts = audio_path.parts

print(f"Audio path: {test_path}")
print(f"Path parts: {path_parts}")

# Extract session info
session_folder = path_parts[path_parts.index("New") + 1]
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

# Encodec path
encodec_path = f"/mnt/msdd/encodec_tokens/{session_folder}/{audio_path.stem}.pt"
print(f"Encodec path: {encodec_path}")
print(f"Encodec exists: {Path(encodec_path).exists()}")

# Conditioning paths
conditioning_roots = [
    f"/mnt/msdd/evenmoreconditioning/{date_folder}/New/{session_folder}/Audio Files",
    f"/mnt/msdd/moreconditioning/{date_folder}/New/{session_folder}/Audio Files"
]

for i, root in enumerate(conditioning_roots):
    print(f"\nConditioning root {i+1}: {root}")
    conditioning_base = f"{root}/{audio_path.stem}"
    test_conditioning = {
        "onsets": f"{conditioning_base}.onsets.npy",
        "rframe": f"{conditioning_base}.rframe.npy",
        "rbend": f"{conditioning_base}.rbend.npy",
        "amp": f"{conditioning_base}.amp.npy",
        "f0": f"{conditioning_base}.f0.npy",
        "f0_masked": f"{conditioning_base}.f0_masked.npy"
    }

    for key, path in test_conditioning.items():
        exists = Path(path).exists()
        print(f"  {key}: {exists} - {path}")