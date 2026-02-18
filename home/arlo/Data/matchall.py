import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Union, Dict, List

# --- CONFIGURATION ---

# Master list of all source audio files
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths5.txt")

# Base directories for all feature types
BASIC_PITCH_DIR = Path("/home/arlo/gcs-bucket/BasicPitch")
LATENT_ROOT = Path("/mnt/msdd/dcae_latentsnew")
FLAT_DIR = LATENT_ROOT / "dcae_latentsnewnew"
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")
CONDITIONING_ROOT = Path("/mnt/msdd/newconditioning")

# --- OUTPUT FILES ---
OUTPUT_JSON = Path("full_dataset.json")
# Log files for missing components
LOG_NO_MIDI = Path("log_no_midi_match.txt")
LOG_NO_LATENT = Path("log_no_latent_match.txt")
LOG_NO_ENCODEC = Path("log_no_encodec_match.txt")
LOG_NO_CONDITIONING = Path("log_no_conditioning_match.txt")


# --- SCRIPT LOGIC ---

# (Helper functions from previous scripts are placed here)

def find_midi_match(audio_path: Path, base_pitch_root: Path) -> Union[str, None]:
    """Finds the corresponding MIDI file."""
    midi_filename = audio_path.with_suffix(".mid").name
    path_parts = audio_path.parts
    
    potential_midi_path = None
    if "protoolsA" in path_parts:
        relative_parts = path_parts[path_parts.index("protoolsA") + 1:]
        potential_midi_path = base_pitch_root.joinpath(*relative_parts).with_name(midi_filename)
    elif "protools" in path_parts:
        relative_parts = path_parts[path_parts.index("protools") + 1:]
        potential_midi_path = base_pitch_root.joinpath("protools", *relative_parts).with_name(midi_filename)
        
    if potential_midi_path and potential_midi_path.exists():
        return str(potential_midi_path)
    return None

def find_latent_match(audio_path: Path, latent_root: Path, flat_dir: Path) -> Union[str, None]:
    """Finds the corresponding DCAE latent .pt file."""
    pt_filename = audio_path.with_suffix(".pt").name
    path_parts = audio_path.parts
    
    try:
        if "protoolsA" in path_parts:
            relative_path = Path(*path_parts[path_parts.index("protoolsA"):])
        elif "protools" in path_parts:
            relative_path = Path(*path_parts[path_parts.index("protools"):])
        else: return None
    except ValueError: return None

    search_paths = [
        latent_root / relative_path.with_name(pt_filename),
        flat_dir / relative_path.with_name(pt_filename)
    ]
    if "protools" in path_parts:
        sub_path_parts = relative_path.parts[1:]
        search_paths.extend([
            latent_root.joinpath(*sub_path_parts).with_name(pt_filename),
            flat_dir.joinpath(*sub_path_parts).with_name(pt_filename)
        ])
    for path in search_paths:
        if path.exists():
            return str(path)
    return None

def find_encodec_match(audio_path: Path, encodec_root: Path) -> Union[str, None]:
    """Finds the corresponding Encodec .pt file."""
    pt_filename = audio_path.with_suffix(".pt").name
    path_parts = audio_path.parts
    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else: return None
        potential_path = encodec_root / session_folder / pt_filename
        if potential_path.exists():
            return str(potential_path)
    except (ValueError, IndexError): return None
    return None

def find_conditioning_matches(audio_path: Path, conditioning_root: Path) -> Union[Dict[str, str], None]:
    """Finds all conditioning source .npy files."""
    path_parts = audio_path.parts
    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else: return None
    except (ValueError, IndexError): return None

    out_base = conditioning_root / session_folder / audio_path.stem
    suffixes = [".onsets.npy", ".rframe.npy", ".rbend.npy", ".amp.npy", ".f0.npy", ".f0_masked.npy"]
    
    found_paths = {}
    for suffix in suffixes:
        path = out_base.with_suffix(suffix)
        if not path.exists():
            return None # If any file is missing, the whole group is invalid
        # Use a short key for the dictionary, e.g., "onsets", "rframe"
        key = suffix.split('.')[1]
        found_paths[key] = str(path)
        
    return found_paths

def main():
    """Main function to orchestrate the entire matching pipeline."""
    if not ALL_AUDIO_PATHS.exists():
        print(f"Error: Master audio list not found at {ALL_AUDIO_PATHS}")
        return

    print(f"Loading master list from {ALL_AUDIO_PATHS}...")
    with open(ALL_AUDIO_PATHS, 'r') as f:
        all_audio_paths = [line.strip() for line in f if line.strip()]

    # --- RESUME LOGIC ---
    existing_data = []
    processed_audio_paths = set()
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r') as f:
            existing_data = json.load(f)
        processed_audio_paths = {item['audio_path'] for item in existing_data}
        print(f"Loaded {len(existing_data)} previously matched entries from {OUTPUT_JSON}")

    # Filter out paths that are already fully processed.
    paths_to_process = [p for p in all_audio_paths if p not in processed_audio_paths]
    print(f"Found {len(paths_to_process)} new or incomplete audio paths to process.")

    newly_matched_data = []
    log_no_midi = []
    log_no_latent = []
    log_no_encodec = []
    log_no_conditioning = []

    for audio_path_str in tqdm(paths_to_process, desc="Matching all features"):
        audio_path = Path(audio_path_str)

        # 1. Match MIDI
        midi_path = find_midi_match(audio_path, BASIC_PITCH_DIR)
        if not midi_path:
            log_no_midi.append(audio_path_str)
            continue
        
        # 2. Match DCAE Latent
        latent_path = find_latent_match(audio_path, LATENT_ROOT, FLAT_DIR)
        if not latent_path:
            log_no_latent.append(audio_path_str)
            continue
            
        # 3. Match Encodec Tokens
        encodec_path = find_encodec_match(audio_path, ENCODEC_ROOT)
        if not encodec_path:
            log_no_encodec.append(audio_path_str)
            continue

        # 4. Match Conditioning Sources
        conditioning_paths = find_conditioning_matches(audio_path, CONDITIONING_ROOT)
        if not conditioning_paths:
            log_no_conditioning.append(audio_path_str)
            continue
            
        # If all successful, create the entry
        entry = {
            "audio_path": audio_path_str,
            "midi_path": midi_path,
            "latent_path": latent_path,
            "encodec_path": encodec_path,
            "conditioning_paths": conditioning_paths
        }
        newly_matched_data.append(entry)

    # --- Save Results and Logs ---
    
    if newly_matched_data:
        print(f"\nFound {len(newly_matched_data)} new complete entries.")
        # Combine old and new data
        final_data = existing_data + newly_matched_data
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"✅ Updated {OUTPUT_JSON.resolve()} with new entries.")
    else:
        print("\nNo new complete entries found.")

    # Write all log files
    if log_no_midi:
        with open(LOG_NO_MIDI, 'w') as f: f.write("\n".join(log_no_midi))
        print(f"❌ Logged {len(log_no_midi)} entries missing a MIDI file to {LOG_NO_MIDI.resolve()}")
    if log_no_latent:
        with open(LOG_NO_LATENT, 'w') as f: f.write("\n".join(log_no_latent))
        print(f"❌ Logged {len(log_no_latent)} entries missing a DCAE latent to {LOG_NO_LATENT.resolve()}")
    if log_no_encodec:
        with open(LOG_NO_ENCODEC, 'w') as f: f.write("\n".join(log_no_encodec))
        print(f"❌ Logged {len(log_no_encodec)} entries missing an Encodec file to {LOG_NO_ENCODEC.resolve()}")
    if log_no_conditioning:
        with open(LOG_NO_CONDITIONING, 'w') as f: f.write("\n".join(log_no_conditioning))
        print(f"❌ Logged {len(log_no_conditioning)} entries missing conditioning files to {LOG_NO_CONDITIONING.resolve()}")


if __name__ == "__main__":
    main()
