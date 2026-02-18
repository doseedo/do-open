import json
import os
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from typing import Union

# --- CONFIGURATION ---

# Input file from your previous script
INPUT_JSON = Path("/home/arlo/Data/midi_matches.json")

# --- NEW: Path to the master list of all audio files ---
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths4.txt") 

# --- Base directories for all feature types ---
LATENT_ROOT = Path("/mnt/msdd/dcae_latentsnew")
FLAT_DIR = LATENT_ROOT / "dcae_latentsnewnew"
ENCODEC_ROOT = Path("/mnt/msdd/encodec_tokens") # NEW

# --- Output file for the new, enriched matches ---
OUTPUT_JSON = Path("encodecfull_matches.json")

# --- SCRIPT LOGIC ---

def find_duplicates_in_flat_dir(flat_dir: Path) -> set:
    """Scans the flat directory to find filenames that appear more than once."""
    if not flat_dir.exists():
        return set()
        
    counts = defaultdict(int)
    for pt_file in flat_dir.glob("*.pt"):
        counts[pt_file.name] += 1
    
    duplicates = {name for name, count in counts.items() if count > 1}
    if duplicates:
        print(f"⚠️ Found {len(duplicates)} duplicate filenames in {flat_dir}. These will be ignored.")
    return duplicates

def find_latent_match(audio_path_str: str, latent_root: Path, flat_dir: Path, duplicates: set) -> Union[str, None]:
    """Finds the corresponding DCAE latent .pt file for a given .wav file."""
    audio_path = Path(audio_path_str)
    pt_filename = audio_path.with_suffix(".pt").name
    path_parts = audio_path.parts
    
    try:
        if "protoolsA" in path_parts:
            relative_path = Path(*path_parts[path_parts.index("protoolsA"):])
        elif "protools" in path_parts:
            relative_path = Path(*path_parts[path_parts.index("protools"):])
        else:
            return None
    except ValueError:
        return None

    # Check all possible locations for the latent file
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

    # Special case for disorganized protoolsA files
    if "protoolsA" in path_parts:
        flat_file_path = flat_dir / pt_filename
        if flat_file_path.exists() and pt_filename not in duplicates:
            tqdm.write(f"  -> Found unstructured file: {pt_filename}. Moving and organizing...")
            try:
                protoolsA_index = path_parts.index("protoolsA")
                sub_path = Path(*path_parts[protoolsA_index + 1:]).parent
                new_destination_dir = flat_dir / "protoolsA" / sub_path
                new_destination_dir.mkdir(parents=True, exist_ok=True)
                new_path = new_destination_dir / pt_filename
                shutil.move(str(flat_file_path), str(new_path))
                tqdm.write(f"     Moved to: {new_path}")
                return str(new_path)
            except Exception as e:
                tqdm.write(f"     ERROR moving file: {e}")
    return None

def find_encodec_match(audio_path_str: str, encodec_root: Path) -> Union[str, None]:
    """Finds the corresponding Encodec .pt file for a given .wav file."""
    audio_path = Path(audio_path_str)
    pt_filename = audio_path.with_suffix(".pt").name
    path_parts = audio_path.parts
    
    try:
        # The session folder is the one after "New" or "Prev"
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


def main():
    """Main function to read midi_matches and find all corresponding feature files."""
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    print("Scanning for duplicate filenames in the flat directory...")
    duplicates_to_ignore = find_duplicates_in_flat_dir(FLAT_DIR)

    with open(INPUT_JSON, 'r') as f:
        midi_matches = json.load(f)

    print(f"Processing {len(midi_matches)} audio/MIDI pairs to find all matching feature files...")

    full_matches_data = []
    midis_without_latent = []
    no_encodec_match = []

    for audio_path, midi_path in tqdm(midi_matches.items(), desc="Matching All Features"):
        latent_path = find_latent_match(audio_path, LATENT_ROOT, FLAT_DIR, duplicates_to_ignore)
        
        if not latent_path:
            midis_without_latent.append(audio_path)
            continue

        encodec_path = find_encodec_match(audio_path, ENCODEC_ROOT)

        if not encodec_path:
            no_encodec_match.append(audio_path)
            continue

        full_matches_data.append({
            "audio_path": audio_path,
            "midi_path": midi_path,
            "latent_path": latent_path,
            "encodec_path": encodec_path
        })

    # --- Save the results ---
    print(f"\nSuccessfully created {len(full_matches_data)} full data entries.")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(full_matches_data, f, indent=4)
    print(f"✅ Full matches saved to: {OUTPUT_JSON.resolve()}")

    if midis_without_latent:
        print(f"\nCould not find latent files for {len(midis_without_latent)} MIDI entries.")
        log_path = Path("midis_without_latent_match.txt")
        with open(log_path, 'w') as f:
            f.write("\n".join(midis_without_latent))
        print(f"❌ List saved to: {log_path.resolve()}")

    if no_encodec_match:
        print(f"\nCould not find Encodec files for {len(no_encodec_match)} entries.")
        log_path = Path("no_encodec_match.txt")
        with open(log_path, 'w') as f:
            f.write("\n".join(no_encodec_match))
        print(f"❌ List saved to: {log_path.resolve()}")


if __name__ == "__main__":
    main()
