#!/usr/bin/env python3
"""
Minimal Vocal Manifest Creator - Efficient version
"""

import json
from pathlib import Path
from tqdm import tqdm
import re

def is_vocal_file(audio_path_str: str) -> bool:
    """Determine if an audio file is vocal-related."""
    path_str = str(audio_path_str).upper()
    filename = Path(audio_path_str).name.upper()

    vocal_patterns = ['VOX_SESS', 'VOX_SESSION', 'VOCAL', 'VOX', '_VOX_', 'LEAD_VOX', 'BACKING_VOX']
    vocal_file_patterns = [r'\bVOX\b', r'\bVOCAL\b', r'\bVOICE\b', r'\bLEAD\s*VOX\b', r'\bBACKING\b', r'\bCHOIR\b']

    for pattern in vocal_patterns:
        if pattern in path_str:
            return True
    for pattern in vocal_file_patterns:
        if re.search(pattern, filename):
            return True
    return False

def find_paths(audio_path: Path):
    """Find all required paths for a vocal file."""
    path_parts = audio_path.parts

    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
            date_folder = path_parts[path_parts.index("New") - 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
            date_folder = path_parts[path_parts.index("Prev") - 1]
        else:
            return None
    except (ValueError, IndexError):
        return None

    # Construct expected paths
    piano_roll_path = f"/mnt/msdd/piano_rolls/{session_folder}/{audio_path.stem}.pianoroll.npy"

    # Latent path - reconstruct the full protools path
    if "protools" in path_parts:
        relative_path = Path(*path_parts[path_parts.index("protools"):])
        latent_path = f"/mnt/msdd/dcae_latentsnew/{relative_path.as_posix()}".replace(".wav", ".pt")
    else:
        return None

    encodec_path = f"/mnt/msdd/encodec_tokens/{session_folder}/{audio_path.stem}.pt"

    # Try different conditioning root locations
    conditioning_roots = [
        f"/mnt/msdd/evenmoreconditioning/{date_folder}/New/{session_folder}/Audio Files",
        f"/mnt/msdd/moreconditioning/{date_folder}/New/{session_folder}/Audio Files"
    ]

    conditioning_paths = None
    for root in conditioning_roots:
        conditioning_base = f"{root}/{audio_path.stem}"
        test_conditioning = {
            "onsets": f"{conditioning_base}.onsets.npy",
            "rframe": f"{conditioning_base}.rframe.npy",
            "rbend": f"{conditioning_base}.rbend.npy",
            "amp": f"{conditioning_base}.amp.npy",
            "f0": f"{conditioning_base}.f0.npy",
            "f0_masked": f"{conditioning_base}.f0_masked.npy"
        }

        # Check if all conditioning files exist
        if all(Path(p).exists() for p in test_conditioning.values()):
            conditioning_paths = test_conditioning
            break

    # Check if all key files exist
    if not Path(piano_roll_path).exists():
        return None
    if not Path(latent_path).exists():
        return None
    if not Path(encodec_path).exists():
        return None
    if not conditioning_paths:
        return None

    group = "vocal"
    sub_group = "lead_vocal" if "LEAD" in str(audio_path).upper() else "vocal_track"

    return {
        "audio_path": str(audio_path),
        "piano_roll_path": piano_roll_path,
        "latent_path": latent_path,
        "encodec_path": encodec_path,
        "conditioning_paths": conditioning_paths,
        "group": group,
        "sub_group": sub_group
    }

def main():
    """Main function."""
    all_audio_paths_file = Path("/home/arlo/Data/all_audio_paths5.txt")
    output_file = Path("/home/arlo/Data/vocal_training_manifest.json")

    print("Loading audio paths...")
    with open(all_audio_paths_file, 'r') as f:
        all_paths = [line.strip() for line in f if line.strip()]

    print(f"Filtering {len(all_paths)} paths for vocals...")
    vocal_paths = [p for p in all_paths if is_vocal_file(p)]
    print(f"Found {len(vocal_paths)} vocal files")

    print("Processing vocal files...")
    vocal_manifest = []

    for path_str in tqdm(vocal_paths[:1000], desc="Processing"):  # Limit to first 1000 for testing
        audio_path = Path(path_str)
        entry = find_paths(audio_path)
        if entry:
            vocal_manifest.append(entry)

    print(f"Created {len(vocal_manifest)} complete entries")

    if vocal_manifest:
        with open(output_file, 'w') as f:
            json.dump(vocal_manifest, f, indent=2)
        print(f"Saved to: {output_file}")

        # Show samples
        for i, entry in enumerate(vocal_manifest[:3]):
            filename = Path(entry['audio_path']).name
            print(f"  {i+1}. {filename} ({entry['group']}/{entry['sub_group']})")

if __name__ == "__main__":
    main()