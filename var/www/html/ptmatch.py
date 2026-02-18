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

# --- NEW: Added a path to the master list of all audio files ---
# This is needed to correctly map latent files back to their original source audio.
ALL_AUDIO_PATHS = Path("/home/arlo/Data/all_audio_paths4.txt") 

# Base directory where the DCAE latent .pt files are stored.
LATENT_ROOT = Path("/mnt/msdd/dcae_latentsnew")

# The specific directory where some protoolsA files are located without structure.
FLAT_DIR = LATENT_ROOT / "dcae_latentsnewnew"

# Output file for the new, enriched matches.
OUTPUT_JSON = Path("full_matches.json")

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
    """
    Finds the corresponding .pt file for a given .wav file, handling multiple path structures
    and the special case for moving disorganized protoolsA files.
    """
    audio_path = Path(audio_path_str)
    pt_filename = audio_path.with_suffix(".pt").name

    # --- Standard Path Construction ---
    path_parts = audio_path.parts
    
    # Check for 'protools' or 'protoolsA' to determine the relative path structure
    try:
        if "protoolsA" in path_parts:
            # Structure: .../protoolsA/DATE/New/...
            relative_path = Path(*path_parts[path_parts.index("protoolsA"):])
        elif "protools" in path_parts:
            # Structure: .../protools/DATE/New/...
            relative_path = Path(*path_parts[path_parts.index("protools"):])
        else:
            return None # Should not happen based on input format
    except ValueError:
        return None

    # --- Search in Possible Locations ---
    
    # Possibility 1: Direct match in the main latent directory with full structure
    potential_path = latent_root / relative_path.with_name(pt_filename)
    if potential_path.exists():
        return str(potential_path)

    # Possibility 1.5 for 'protools' files
    if "protools" in path_parts:
        sub_path_parts = relative_path.parts[1:] # Skips 'protools'
        potential_path_alt_root = latent_root.joinpath(*sub_path_parts).with_name(pt_filename)
        if potential_path_alt_root.exists():
            return str(potential_path_alt_root)

    # Possibility 2: Match inside the 'dcae_latentsnewnew' subdirectory with full structure
    potential_path = flat_dir / relative_path.with_name(pt_filename)
    if potential_path.exists():
        return str(potential_path)

    # Possibility 3: Match inside 'dcae_latentsnewnew' without the 'protools' prefix
    if "protools" in path_parts:
        sub_path_parts = relative_path.parts[1:] # Skips 'protools'
        potential_path_alt_flat = flat_dir.joinpath(*sub_path_parts).with_name(pt_filename)
        if potential_path_alt_flat.exists():
            return str(potential_path_alt_flat)

    # --- Special Case: Unstructured protoolsA files in the flat directory ---
    if "protoolsA" in path_parts:
        flat_file_path = flat_dir / pt_filename
        
        if flat_file_path.exists():
            if pt_filename in duplicates:
                tqdm.write(f"  - Ignoring duplicate file found in flat directory: {pt_filename}")
                return None

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

    return None

def main():
    """
    Main function to read midi_matches, find latent files, and create a new unified JSON.
    """
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return
    if not ALL_AUDIO_PATHS.exists():
        print(f"Error: Master audio list not found at {ALL_AUDIO_PATHS}")
        return

    print("Scanning for duplicate filenames in the flat directory...")
    duplicates_to_ignore = find_duplicates_in_flat_dir(FLAT_DIR)

    with open(INPUT_JSON, 'r') as f:
        midi_matches = json.load(f)

    print(f"Processing {len(midi_matches)} audio/MIDI pairs to find matching latent files...")

    full_matches_data = []
    midis_without_latent = []
    
    for audio_path, midi_path in tqdm(midi_matches.items(), desc="Matching Latent Files"):
        latent_path = find_latent_match(audio_path, LATENT_ROOT, FLAT_DIR, duplicates_to_ignore)
        
        if latent_path:
            full_matches_data.append({
                "audio_path": audio_path,
                "midi_path": midi_path,
                "latent_path": latent_path
            })
        else:
            midis_without_latent.append(audio_path)

    # --- MODIFIED: Find latents that were never matched to a MIDI file ---
    print("\nScanning all latent files to find ones without MIDI matches...")
    all_latent_paths = list(LATENT_ROOT.rglob("*.pt"))
    latents_without_midi_source_paths = []
    
    # --- FIX: Build a complete map from the master audio list ---
    print(f"Building a complete map of all source audio files from {ALL_AUDIO_PATHS}...")
    with open(ALL_AUDIO_PATHS, 'r') as f:
        all_source_paths = [line.strip() for line in f if line.strip()]
    stem_to_source_path = {Path(p).stem: p for p in all_source_paths}
    
    # Create a set of stems from audio files that had a MIDI match for efficient lookup
    audio_stems_with_midi = {Path(p).stem for p in midi_matches.keys()}

    for latent_path in tqdm(all_latent_paths, desc="Checking for unused latents"):
        # If a latent's stem doesn't have a corresponding MIDI entry...
        if latent_path.stem not in audio_stems_with_midi:
            # ...find the original source audio path for that stem from the complete map.
            source_path = stem_to_source_path.get(latent_path.stem)
            if source_path:
                latents_without_midi_source_paths.append(source_path)

    # --- Save the results ---
    print(f"\nSuccessfully found and linked {len(full_matches_data)} latent files.")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(full_matches_data, f, indent=4)
    print(f"✅ Full matches saved to: {OUTPUT_JSON.resolve()}")

    if midis_without_latent:
        print(f"\nCould not find latent files for {len(midis_without_latent)} MIDI entries.")
        no_match_log = Path("midis_without_latent_match.txt")
        with open(no_match_log, 'w') as f:
            for path in midis_without_latent:
                f.write(f"{path}\n")
        print(f"❌ List of MIDI entries without a latent saved to: {no_match_log.resolve()}")

    if latents_without_midi_source_paths:
        print(f"\nFound {len(latents_without_midi_source_paths)} latent files that do not have a MIDI match.")
        no_match_log_latents = Path("latents_without_midi_match.txt")
        with open(no_match_log_latents, 'w') as f:
            for path in latents_without_midi_source_paths:
                f.write(f"{path}\n")
        print(f"❌ List of source audio for latents without a MIDI entry saved to: {no_match_log_latents.resolve()}")


if __name__ == "__main__":
    main()
