import json
from pathlib import Path
from tqdm import tqdm
from typing import Optional

# --- CONFIGURATION ---

INPUT_JSON = Path("full_dataset_new.json") 
PIANO_ROLL_ROOT = Path("/mnt/msdd/piano_rolls")
FINAL_MANIFEST_JSON = Path("full_dataset_match.json")
LOG_NO_PIANO_ROLL = Path("log_no_piano_roll_match.txt")


def find_piano_roll_match(audio_path: Path, piano_roll_root: Path) -> Optional[str]:
    pr_filename = audio_path.with_suffix(".pianoroll.npy").name
    path_parts = audio_path.parts

    try:
        if "New" in path_parts:
            session_folder = path_parts[path_parts.index("New") + 1]
        elif "Prev" in path_parts:
            session_folder = path_parts[path_parts.index("Prev") + 1]
        else:
            session_folder = path_parts[path_parts.index("2025-05-11") + 1]

        potential_pr_path = piano_roll_root / session_folder / pr_filename
        if potential_pr_path.exists():
            return str(potential_pr_path)

    except (ValueError, IndexError):
        pass

    # Fallback: root search
    potential_pr_path_root = piano_roll_root / pr_filename
    if potential_pr_path_root.exists():
        return str(potential_pr_path_root)

    return None


def main():
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    print(f"Loading master list from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        all_entries = json.load(f)

    existing_data = []
    processed_audio_paths = set()
    if FINAL_MANIFEST_JSON.exists():
        with open(FINAL_MANIFEST_JSON, 'r') as f:
            existing_data = json.load(f)
        processed_audio_paths = {item['audio_path'] for item in existing_data}
        print(f"Loaded {len(existing_data)} previously matched entries")

    entries_to_process = [e for e in all_entries if e['audio_path'] not in processed_audio_paths]
    print(f"Found {len(entries_to_process)} new entries to process.")

    newly_matched_data = []
    log_no_piano_roll = []

    for entry in tqdm(entries_to_process, desc="Matching Piano Rolls"):
        audio_path_str = entry['audio_path']
        audio_path = Path(audio_path_str)

        piano_roll_path = find_piano_roll_match(audio_path, PIANO_ROLL_ROOT)

        if not piano_roll_path:
            log_no_piano_roll.append(audio_path_str)

        new_entry = {
            "audio_path": audio_path_str,
            "piano_roll_path": piano_roll_path  # <-- stays None if not found
        }
        for key, value in entry.items():
            if key not in ["audio_path", "midi_path"]:
                new_entry[key] = value

        newly_matched_data.append(new_entry)

    if newly_matched_data:
        final_data = existing_data + newly_matched_data
        with open(FINAL_MANIFEST_JSON, 'w') as f:
            json.dump(final_data, f, indent=4)
        print(f"✅ Updated {FINAL_MANIFEST_JSON} with {len(newly_matched_data)} entries.")

    if log_no_piano_roll:
        with open(LOG_NO_PIANO_ROLL, 'w') as f:
            f.write("\n".join(log_no_piano_roll))
        print(f"❌ Logged {len(log_no_piano_roll)} missing piano roll entries to {LOG_NO_PIANO_ROLL}")


if __name__ == "__main__":
    main()
