import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# --- CONFIGURATION ---

# The JSON file containing your matched feature paths.
INPUT_JSON = Path("full_dataset_match.json") 

# The root directory containing your categorized .txt files.
CATEGORIZED_PATHS_DIR = Path("/home/arlo/Data/categorized_instrument_paths_subcats_lists")

# The final, enriched JSON file that your training script will use.
FINAL_MANIFEST_JSON = Path("final_training_manifest.json")

# --- FILTERING CONTROLS ---
# Set to True to only include instruments from the APPROVED_GROUPS list.
FILTER_BY_TYPE = True 
# List of instrument groups to include when FILTER_BY_TYPE is True.
APPROVED_GROUPS = [
    "piano",
    "guitar",
    "bass",
    "strings",
    "brass",
    "winds"
]

# --- NEW: KEYWORD-BASED SKIPPING ---
# Any entry whose audio_path contains one of these (case-insensitive) words will be skipped.
SKIP_KEYWORDS = [
    "delay",
    "verb",
    "reverb"

]


# --- SCRIPT LOGIC ---

def build_category_map(categorized_dir: Path) -> dict:
    """
    Scans the categorized .txt files to build a map from an audio path
    to its group and subgroup.
    """
    print(f"Building category map from: {categorized_dir}")
    path_to_category = {}
    
    # Find all .txt files, which can be in subdirectories (group/subgroup.txt)
    for txt_file in tqdm(list(categorized_dir.rglob("*.txt")), desc="Reading category files"):
        sub_group = txt_file.stem
        group = txt_file.parent.name
        
        if group == categorized_dir.name:
            group = sub_group
            sub_group = None
            
        with open(txt_file, 'r') as f:
            for line in f:
                audio_path = line.strip()
                if audio_path:
                    path_to_category[audio_path] = {"group": group, "sub_group": sub_group}
                    
    return path_to_category

def main():
    """
    Main function to enrich the dataset manifest with instrument group and type.
    """
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return
    if not CATEGORIZED_PATHS_DIR.exists():
        print(f"Error: Categorized paths directory not found at {CATEGORIZED_PATHS_DIR}")
        return

    # 1. Build a lookup map from the categorized .txt files
    category_map = build_category_map(CATEGORIZED_PATHS_DIR)
    print(f"Mapped {len(category_map)} audio files to their instrument categories.")

    # 2. Load the existing manifest with all the feature paths
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)
    
    # --- NEW: Pre-filter based on SKIP_KEYWORDS ---
    if SKIP_KEYWORDS:
        print(f"Filtering entries based on {len(SKIP_KEYWORDS)} skip keywords...")
        initial_count = len(data)
        # Create a lowercase version of the keywords for case-insensitive matching
        lower_skip_keywords = [k.lower() for k in SKIP_KEYWORDS]
        
        filtered_entries = [
            entry for entry in data 
            if not any(keyword in entry['audio_path'].lower() for keyword in lower_skip_keywords)
        ]
        skipped_count = initial_count - len(filtered_entries)
        print(f"Skipped {skipped_count} entries containing keywords.")
        data = filtered_entries
    # --- END NEW ---

    # 3. Create the new, enriched manifest
    final_manifest = []
    unmapped_count = 0
    filtered_out_count = 0
    
    print(f"Enriching {len(data)} entries with category information...")
    for entry in tqdm(data, desc="Enriching Manifest"):
        audio_path = entry["audio_path"]
        
        if audio_path in category_map:
            category_info = category_map[audio_path]
            
            # --- FILTERING LOGIC ---
            if FILTER_BY_TYPE and category_info.get("group") not in APPROVED_GROUPS:
                filtered_out_count += 1
                continue # Skip this entry if its group is not in the approved list
            # --- END FILTERING LOGIC ---

            # Add the group and sub_group keys to the entry
            entry.update(category_info)
            final_manifest.append(entry)
        else:
            unmapped_count += 1

    # 4. Save the final result
    print(f"\nSuccessfully enriched {len(final_manifest)} entries.")
    if FILTER_BY_TYPE:
        print(f"Filtered out {filtered_out_count} entries based on APPROVED_GROUPS list.")
    if unmapped_count > 0:
        print(f"⚠️  Warning: {unmapped_count} entries from your JSON could not be found in the category .txt files and were skipped.")
        
    with open(FINAL_MANIFEST_JSON, 'w') as f:
        json.dump(final_manifest, f, indent=4)
        
    print(f"✅ Final training manifest saved to: {FINAL_MANIFEST_JSON.resolve()}")


if __name__ == "__main__":
    main()
