import json
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict

# --- CONFIGURATION ---

# Input file from your MIDI matching script
INPUT_JSON = Path("/home/arlo/Data/midi_matches.json")

# Base directories for all feature types
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")
CONDITIONING_ROOT = Path("/mnt/msdd/newconditioning")

# --- OUTPUT FILES ---
OUTPUT_JSON = Path("full_matches_with_conditioning.json")
# Log files for missing components
LOG_NO_ENCODEC = Path("log_no_encodec_match.txt")
LOG_NO_CONDITIONING = Path("log_no_conditioning_match.txt")


# --- SCRIPT LOGIC ---

def find_encodec_match(audio_path: Path, encodec_root: Path) -> Union[str, None]:
    """Finds the corresponding Encodec .pt file for a given .wav file."""
    pt_filename = audio_path.with_suffix(".pt").name
    path_parts = audio_path.parts
    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else:
            return None
        potential_path = encodec_root / session_folder / pt_filename
        if potential_path.exists():
            return str(potential_path)
    except (ValueError, IndexError):
        return None
    return None

def find_conditioning_matches(audio_path: Path, conditioning_root: Path) -> Union[Dict[str, str], None]:
    """Finds all conditioning source .npy files."""
    path_parts = audio_path.parts
    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else:
            return None
    except (ValueError, IndexError):
        return None

    out_base = conditioning_root / session_folder / audio_path.stem
    suffixes = [".onsets.npy", ".rframe.npy", ".rbend.npy", ".amp.npy", ".f0.npy", ".f0_masked.npy"]
    
    found_paths = {}
    for suffix in suffixes:
        path = out_base.with_suffix(suffix)
        if not path.exists():
            # If any single file is missing, the entire group is considered invalid for training
            return None
        # Use a short key for the dictionary, e.g., "onsets", "rframe"
        key = suffix.split('.')[1]
        found_paths[key] = str(path)
        
    return found_paths

def main():
    """Main function to orchestrate the matching pipeline."""
    if not INPUT_JSON.exists():
        print(f"Error: Input MIDI matches file not found at {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r') as f:
        midi_matches = json.load(f)

    # --- RESUME LOGIC ---
    existing_data = []
    processed_audio_paths = set()
    if OUTPUT_JSON.exists():
        with open(OUTPUT_JSON, 'r') as f:
            existing_data = json.load(f)
        processed_audio_paths = {item['audio_path'] for item in existing_data}
        print(f"Loaded {len(existing_data)} previously matched entries from {OUTPUT_JSON}")

    # Filter out entries that are already fully processed.
    entries_to_process = {
        audio_path: midi_path for audio_path, midi_path in midi_matches.items()
        if audio_path not in processed_audio_paths
    }
    print(f"Found {len(entries_to_process)} new or incomplete audio/MIDI pairs to process.")

    newly_matched_data = []
    log_no_encodec = []
    log_no_conditioning = []

    for audio_path_str, midi_path in tqdm(entries_to_process.items(), desc="Matching Features"):
        audio_path = Path(audio_path_str)
            
        # 1. Match Encodec Tokens
        encodec_path = find_encodec_match(audio_path, ENCODEC_ROOT)
        if not encodec_path:
            log_no_encodec.append(audio_path_str)
            continue

        # 2. Match Conditioning Sources
        conditioning_paths = find_conditioning_matches(audio_path, CONDITIONING_ROOT)
        if not conditioning_paths:
            log_no_conditioning.append(audio_path_str)
            continue
            
        # If all successful, create the entry
        entry = {
            "audio_path": audio_path_str,
            "midi_path": midi_path,
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
    if log_no_encodec:
        with open(LOG_NO_ENCODEC, 'w') as f: f.write("\n".join(log_no_encodec))
        print(f"❌ Logged {len(log_no_encodec)} entries missing an Encodec file to {LOG_NO_ENCODEC.resolve()}")
    if log_no_conditioning:
        with open(LOG_NO_CONDITIONING, 'w') as f: f.write("\n".join(log_no_conditioning))
        print(f"❌ Logged {len(log_no_conditioning)} entries missing conditioning files to {LOG_NO_CONDITIONING.resolve()}")


if __name__ == "__main__":
    main()
