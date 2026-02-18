import json
from pathlib import Path
from tqdm import tqdm
from typing import Union, Dict, Optional

# --- CONFIGURATION ---

# Input file from your MIDI matching script
INPUT_JSON = Path("/home/arlo/Data/midi_matches.json")

# Base directories for all feature types
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens")
CONDITIONING_ROOT = Path("/mnt/msdd/newconditioning")

# --- OUTPUT FILES ---
OUTPUT_JSON = Path("full_matches_with_conditioning.json")
# Log file for any entry that has at least one missing component
LOG_INCOMPLETE_MATCHES = Path("log_incomplete_matches.txt")


# --- SCRIPT LOGIC ---

def find_encodec_match(audio_path: Path, encodec_root: Path) -> Optional[str]:
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

def find_conditioning_matches(audio_path: Path, conditioning_root: Path) -> Dict[str, Optional[str]]:
    """
    Finds all available conditioning source .npy files.
    Returns a dictionary with paths or None for missing files.
    """
    path_parts = audio_path.parts
    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else:
            # If no session folder, we can't construct the path
            return {
                "onsets": None, "rframe": None, "rbend": None, 
                "amp": None, "f0": None, "f0_masked": None
            }
    except (ValueError, IndexError):
        return {
            "onsets": None, "rframe": None, "rbend": None, 
            "amp": None, "f0": None, "f0_masked": None
        }

    out_base = conditioning_root / session_folder / audio_path.stem
    suffixes = [".onsets.npy", ".rframe.npy", ".rbend.npy", ".amp.npy", ".f0.npy", ".f0_masked.npy"]
    
    found_paths = {}
    for suffix in suffixes:
        path = out_base.with_suffix(suffix)
        key = suffix.split('.')[1]
        if path.exists():
            found_paths[key] = str(path)
        else:
            # If a file is missing, record its path as None
            found_paths[key] = None
            
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
    log_incomplete = []

    for audio_path_str, midi_path in tqdm(entries_to_process.items(), desc="Matching Features"):
        audio_path = Path(audio_path_str)
            
        # 1. Match Encodec Tokens (returns path or None)
        encodec_path = find_encodec_match(audio_path, ENCODEC_ROOT)

        # 2. Match Conditioning Sources (returns a dict with paths or None)
        conditioning_paths = find_conditioning_matches(audio_path, CONDITIONING_ROOT)
            
        # Create the entry regardless of whether all files were found
        entry = {
            "audio_path": audio_path_str,
            "midi_path": midi_path,
            "encodec_path": encodec_path,
            "conditioning_paths": conditioning_paths
        }
        newly_matched_data.append(entry)

        # Log if any of the crucial paths are missing
        if not encodec_path or not all(conditioning_paths.values()):
            log_incomplete.append(audio_path_str)

    # --- Save Results and Logs ---
    
    if newly_matched_data:
        print(f"\nProcessed {len(newly_matched_data)} new entries.")
        # Combine old and new data
        final_data = existing_data + newly_matched_data
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"✅ Updated {OUTPUT_JSON.resolve()} with new entries.")
    else:
        print("\nNo new entries to process.")

    # Write a single log file for all incomplete entries
    if log_incomplete:
        with open(LOG_INCOMPLETE_MATCHES, 'w') as f:
            f.write("\n".join(log_incomplete))
        print(f"❌ Logged {len(log_incomplete)} entries with one or more missing feature files to {LOG_INCOMPLETE_MATCHES.resolve()}")


if __name__ == "__main__":
    main()
