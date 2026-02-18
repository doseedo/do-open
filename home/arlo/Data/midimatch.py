import json
from pathlib import Path
from tqdm import tqdm
from typing import Union

# --- CONFIGURATION ---

# Input file containing the list of processed .wav files.
PROGRESS_LOG = Path("/mnt/msdd/dcae_latentsnew/progress_log.txt")

# Base directory where the BasicPitch MIDI files are stored.
BASIC_PITCH_DIR = Path("/home/arlo/gcs-bucket/BasicPitch")

# Output files
MATCHES_JSON = Path("midi_matchesnew.json")
NO_MATCHES_LOG = Path("midi_no_matches.txt")

# --- SCRIPT LOGIC ---

def find_midi_match(audio_path_str: str, base_pitch_root: Path) -> Union[str, None]:
    """
    Takes a single audio path and tries to find a corresponding MIDI file.
    It handles the two different path structures for 'protools' and 'protoolsA'.
    """
    audio_path = Path(audio_path_str)
    
    # The expected MIDI file will have the same name but with a .mid extension
    midi_filename = audio_path.with_suffix(".mid").name

    # --- Case 1: Path is in 'protoolsA' ---
    # Example: /home/arlo/gcs-bucket/protoolsA/.../file.wav -> /home/arlo/gcs-bucket/BasicPitch/.../file.mid
    if "protoolsA" in audio_path.parts:
        # Reconstruct the path by replacing the 'protoolsA' part
        relative_parts = audio_path.parts[audio_path.parts.index("protoolsA") + 1:]
        potential_midi_path = base_pitch_root.joinpath(*relative_parts).with_name(midi_filename)
        if potential_midi_path.exists():
            return str(potential_midi_path)

    # --- Case 2: Path is in 'protools' ---
    # Example: /home/arlo/gcs-bucket/protools/.../file.wav -> /home/arlo/gcs-bucket/BasicPitch/protools/.../file.mid
    elif "protools" in audio_path.parts:
        # Reconstruct the path by adding 'protools' after the BasicPitch root
        relative_parts = audio_path.parts[audio_path.parts.index("protools") + 1:]
        potential_midi_path = base_pitch_root.joinpath("protools", *relative_parts).with_name(midi_filename)
        if potential_midi_path.exists():
            return str(potential_midi_path)
            
    # If neither case matches or the file doesn't exist
    return None

def main():
    """
    Main function to read the progress log, find matches, and write output files.
    """
    if not PROGRESS_LOG.exists():
        print(f"Error: Progress log not found at {PROGRESS_LOG}")
        return

    print(f"Reading audio paths from {PROGRESS_LOG}...")
    with open(PROGRESS_LOG, 'r') as f:
        audio_paths = [line.strip() for line in f if line.strip()]

    print(f"Found {len(audio_paths)} paths. Searching for MIDI matches in {BASIC_PITCH_DIR}...")

    matches = {}
    no_matches = []

    for audio_path in tqdm(audio_paths, desc="Finding MIDI Matches"):
        midi_path = find_midi_match(audio_path, BASIC_PITCH_DIR)
        
        if midi_path:
            matches[audio_path] = midi_path
        else:
            no_matches.append(audio_path)

    # --- Save the results ---
    
    # Save the successful matches to a JSON file
    print(f"\nFound {len(matches)} successful matches.")
    with open(MATCHES_JSON, 'w') as f:
        json.dump(matches, f, indent=4)
    print(f"✅ Matches saved to: {MATCHES_JSON.resolve()}")

    # Save the paths with no matches to a text file
    print(f"\nFound {len(no_matches)} paths with no match.")
    with open(NO_MATCHES_LOG, 'w') as f:
        for path in no_matches:
            f.write(f"{path}\n")
    print(f"❌ Unmatched paths saved to: {NO_MATCHES_LOG.resolve()}")


if __name__ == "__main__":
    main()
