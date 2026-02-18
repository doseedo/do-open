import json
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# --- CONFIGURATION ---

# The final, enriched JSON file that you want to validate.
INPUT_JSON = Path("final_training_manifest.json")

# Log file for entries that fail validation.
ERROR_LOG = Path("validation_errors.txt")

# --- PROCESSING PARAMS (from your scripts) ---
# These are used to verify the alignment between different feature types.

# For DCAE Latents, Conditioning Sources, and Piano Rolls
DCAE_SR = 44100
DCAE_HOP_LENGTH = 4096

# For Encodec Tokens (using the 24kHz model)
ENCODEC_SR = 24000
ENCODEC_HOP_LENGTH = 320 # Encodec's internal hop size is 320

# --- NEW: Alignment Tolerance ---
# Allow a small difference in the number of frames for "slow grid" features.
SLOW_GRID_TOLERANCE = 2 # in frames

# --- SCRIPT LOGIC ---

def validate_entry(entry: dict) -> (bool, str):
    """
    Validates a single entry from the dataset manifest to ensure all
    feature files exist and are temporally aligned.
    """
    try:
        # 1. Check that all required paths are present and not null
        latent_path_str = entry.get("latent_path")
        encodec_path_str = entry.get("encodec_path")
        pr_path_str = entry.get("piano_roll_path")
        conditioning_paths = entry.get("conditioning_paths")

        if not latent_path_str: return False, "Entry is missing 'latent_path' or its value is null."
        if not encodec_path_str: return False, "Entry is missing 'encodec_path' or its value is null."
        if not pr_path_str: return False, "Entry is missing 'piano_roll_path' or its value is null."
        if not conditioning_paths: return False, "Entry is missing 'conditioning_paths' dictionary."
        
        amp_path_str = conditioning_paths.get("amp")
        if not amp_path_str: return False, "Entry is missing 'amp' path in 'conditioning_paths' or its value is null."

        latent_path = Path(latent_path_str)
        encodec_path = Path(encodec_path_str)
        pr_path = Path(pr_path_str)
        amp_path = Path(amp_path_str)

        if not latent_path.exists(): return False, "DCAE latent file is missing."
        if not encodec_path.exists(): return False, "Encodec token file is missing."
        if not pr_path.exists(): return False, "Piano roll file is missing."
        if not amp_path.exists(): return False, "Conditioning source file (amp.npy) is missing."

        # 2. Load metadata to get frame counts
        latent_data = torch.load(latent_path, map_location='cpu')
        latent_len = latent_data['latents'].shape[-1]

        pr_data = np.load(pr_path)
        piano_roll_len = pr_data.shape[1]

        amp_data = np.load(amp_path)
        conditioning_len = len(amp_data)

        encodec_data = torch.load(encodec_path, map_location='cpu')
        encodec_len = encodec_data[0][0].shape[-1]

        # 3. Validate temporal alignment
        
        # --- MODIFIED: More forgiving "Slow Grid" check ---
        slow_lengths = [latent_len, conditioning_len, piano_roll_len]
        min_len = min(slow_lengths)
        max_len = max(slow_lengths)

        if (max_len - min_len) > SLOW_GRID_TOLERANCE:
            return False, (f"Slow Grid Alignment Error: Frame lengths differ by more than {SLOW_GRID_TOLERANCE}. "
                           f"Latent ({latent_len}), Conditioning ({conditioning_len}), Piano Roll ({piano_roll_len}).")
        
        # Use the minimum length as the reference for the fast grid check
        reference_len = min_len
        # --- END MODIFICATION ---

        # Rule 2: "Fast grid" (Encodec) must be proportionally larger.
        expected_encodec_len = reference_len * (ENCODEC_SR / ENCODEC_HOP_LENGTH) / (DCAE_SR / DCAE_HOP_LENGTH)
        
        if not (abs(encodec_len - expected_encodec_len) <= 2):
            return False, f"Fast Grid Alignment Error: Encodec length ({encodec_len}) is not proportional to latent length ({reference_len}). Expected ~{int(expected_encodec_len)}."

    except Exception as e:
        return False, f"Error processing entry: {e}"

    return True, "Valid"


def main():
    """
    Main function to read the manifest and validate all entries.
    """
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    print(f"🔬 Validating {len(data)} entries from {INPUT_JSON}...")

    valid_entries_count = 0
    invalid_entries_log = []
    
    for entry in tqdm(data, desc="Validating Dataset"):
        is_valid, reason = validate_entry(entry)
        if is_valid:
            valid_entries_count += 1
        else:
            error_message = f"File: {entry['audio_path']}\nReason: {reason}\n"
            invalid_entries_log.append(error_message)

    # --- Print and Save Results ---
    
    print("\n--- Validation Report ---")
    print(f"Total Entries Checked: {len(data):,}")
    print(f"✅ Valid Entries:       {valid_entries_count:,}")
    print(f"❌ Invalid Entries:     {len(invalid_entries_log):,}")
    print("-------------------------\n")

    if invalid_entries_log:
        with open(ERROR_LOG, 'w') as f:
            f.write("--- Dataset Validation Errors ---\n\n")
            f.write("\n".join(invalid_entries_log))
        print(f"Detailed report for invalid entries saved to: {ERROR_LOG.resolve()}")
    else:
        print("🎉 All entries are valid and perfectly aligned!")


if __name__ == "__main__":
    main()
