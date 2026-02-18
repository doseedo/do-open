import multiprocessing as mp
import os
import torchaudio
import torch
from pathlib import Path
from tqdm import tqdm
import gc
import re
import traceback

# --- Multiprocessing setup ---
# This is necessary for CUDA to work correctly with multiprocessing
# if you were to re-introduce GPU-dependent code.
mp.set_start_method("spawn", force=True)

# --- Configuration & Exclusions ---
# Keywords to exclude, assuming these are drum tracks that may be silent.
exclude_keywords = [
    "Kick", "KICK", "KickIn", "KickOut", "KickSub", "Kik", "Kck", "KIK", "kick", "kik",
    "Snare", "Snr", "SNR", "Sn", "SN", "SnrTop", "SnrBtm", "SNARE", "snare", "sn",
    "HiHat", "HH", "ClosedHat", "OpenHat", "Hat", "HAT", "hihat",
    "Tom", "RackTom", "FloorTom", "RTom", "FTom", "TOM",
    "Cymbal", "Cym", "Crash", "Ride", "Splash", "China", "Stack", "CYM", "CYMBAL",
    "OH", "Overhead", "OHL", "OHR", "OVERHEAD",
    "Perc", "Tamb", "Cowbell", "Clap", "Shaker", "Triangle", "PERC", "CLAP",
    "Drum", "Drums", "Drumkit", "Kit", "KIT", "DRUM", "DRUMKIT"
]
exclude_pattern = re.compile(r"|".join(re.escape(word) for word in exclude_keywords), re.IGNORECASE)

# Silence detection parameters
SILENCE_THRESHOLD = 1e-6  # Minimum amplitude to consider non-silent
SILENCE_DURATION_S = 5    # Minimum duration of silence to be trimmed

# === SCRIPT-SPECIFIC CONFIGURATION ===
# The path to the text file containing a list of audio file paths.
# Update this to your file's location.
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths3.txt")

# The directory where the trimmed files will be saved.
# This will be created in the same directory as this script.
SAVE_DIR = Path(__file__).parent / "trimmed_audio"
SAVE_DIR.mkdir(exist_ok=True)


# --- Worker Function ---
def worker(worker_id, paths, input_dir, output_dir):
    """
    Worker function to process a chunk of audio files.
    """
    success_count = 0
    skipped_count = 0
    failed_count = 0

    for wav_path in tqdm(paths, desc=f"Worker {worker_id}"):
        try:
            # Check for exclusion keywords in the file path
            if exclude_pattern.search(str(wav_path)):
                print(f"⏭ [Worker {worker_id}] Skipped (drum file): {wav_path}")
                skipped_count += 1
                continue

            # Load the audio file
            waveform, sr = torchaudio.load(wav_path)
            
            # --- Silence Trimming Logic ---
            # To handle mono/stereo, calculate the energy across all channels
            audio_energy = torch.mean(waveform**2, dim=0)
            
            # Define a silence threshold in terms of energy (square of amplitude threshold)
            energy_threshold = SILENCE_THRESHOLD**2
            
            # Find silent sections (where energy is below the threshold for 5 seconds or more)
            silent_sections = (audio_energy < energy_threshold).float()
            
            # Convert 5 seconds to number of samples
            silent_samples = int(SILENCE_DURATION_S * sr)
            
            # Use a convolution to find contiguous sections of silence
            kernel = torch.ones(silent_samples)
            silent_sum = torch.conv1d(silent_sections.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=silent_samples//2).squeeze()
            
            # Find the start and end indices of the first and last non-silent segments
            # We look for where the silent_sum is less than the total silent samples,
            # meaning there's a non-silent part in that window.
            non_silent_indices = torch.where(silent_sum < silent_samples)[0]
            
            if len(non_silent_indices) == 0:
                # The entire file is considered silent, so we check if its max amplitude is
                # below the general threshold and skip it if it is.
                if waveform.abs().max() < SILENCE_THRESHOLD:
                    print(f"⏭ [Worker {worker_id}] Skipped (entirely silent): {wav_path}")
                    skipped_count += 1
                    continue
                # If not entirely silent, we don't trim.
                start_trim = 0
                end_trim = waveform.shape[-1]
            else:
                start_trim = non_silent_indices[0]
                end_trim = non_silent_indices[-1]

            # Compare trimmed audio length to original
            original_length = waveform.shape[-1]
            trimmed_length = end_trim - start_trim
            
            if trimmed_length < original_length:
                trimmed_waveform = waveform[:, start_trim:end_trim]

                # Create the output path, preserving the subdirectory structure
                relative_path = Path(wav_path).relative_to(input_dir)
                output_path = Path(output_dir) / relative_path
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the trimmed audio
                torchaudio.save(output_path, trimmed_waveform, sr)
                
                success_count += 1
                print(f"✅ [Worker {worker_id}] Trimmed and saved: {output_path}")
                print(f"  - Original duration: {original_length/sr:.2f}s, Trimmed: {trimmed_length/sr:.2f}s")
            else:
                skipped_count += 1
                print(f"⏭ [Worker {worker_id}] Skipped (no silence to trim): {wav_path}")

        except Exception as e:
            failed_count += 1
            print(f"❌ [Worker {worker_id}] Failed: {wav_path} | Error: {e}")
            traceback.print_exc()
        finally:
            gc.collect()

    return success_count, skipped_count, failed_count

# --- Main function ---
def main():
    # Load all audio paths from the specified file
    with open(AUDIO_LIST_FILE, 'r') as f:
        audio_paths = [Path(line.strip()) for line in f if line.strip().endswith(".wav")]
    
    if not audio_paths:
        print(f"No WAV files found in {AUDIO_LIST_FILE}")
        return

    # Infer the common base directory from the first audio path
    input_dir = audio_paths[0].parent
    
    print(f"🚀 Found {len(audio_paths)} WAV files from {AUDIO_LIST_FILE}. Processing...")
    
    num_processes = mp.cpu_count()
    chunks = [audio_paths[i::num_processes] for i in range(num_processes)]

    processes = []
    
    with mp.Manager() as manager:
        success_list = manager.list()
        skipped_list = manager.list()
        failed_list = manager.list()

        for i in range(num_processes):
            if chunks[i]:
                p = mp.Process(target=worker, args=(i, chunks[i], input_dir, SAVE_DIR))
                processes.append(p)
                p.start()

        for p in processes:
            p.join()
    
    print("\n✅ All workers finished.")
    print(f"--- Summary ---")
    print(f"✅ Trimmed and saved: {len(success_list)}")
    print(f"⏭ Skipped: {len(skipped_list)}")
    print(f"❌ Failed: {len(failed_list)}")

if __name__ == "__main__":
    main()
