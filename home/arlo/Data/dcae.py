import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import os
import torchaudio
import torch
from acestep.pipeline_ace_step import ACEStepPipeline
from pathlib import Path
from tqdm import tqdm
import gc
import json
import numpy as np
from collections import defaultdict
import re
import traceback
from multiprocessing import Process


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

# === CONFIGURATION ===
AUDIO_LIST_FILE = Path("/home/arlo/Data/all_audio_paths6.txt")
SAVE_DIR = Path("/mnt/msdd/dcae_latentsnew")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
# --- CHANGE: Corrected log paths and changed error log to .jsonl ---
PROGRESS_LOG = SAVE_DIR / "progress_log.txt"
ERROR_LOG = SAVE_DIR / "error_log.jsonl" # Use JSON Lines for safe concurrent writing
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STD_FAIL_LOG = SAVE_DIR / "std_failures.jsonl" # This is still unused

# Model parameters
LATENT_SHAPE = (8, 16)
DOWNSAMPLE_FACTOR = 4096
MIN_SAMPLES = int(3 * 48000)
MAX_SAMPLES = int(12 * 60 * 48000)

# Quality thresholds
SILENCE_THRESHOLD = 1e-6
STD_RANGE = (0.4, 2.2)
MEAN_RANGE = (-1.3, 1.3)
DURATION_TOLERANCE = 0.05

# === GPU SETUP ===
NUM_GPUS = torch.cuda.device_count()
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, range(NUM_GPUS)))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def gpu_worker(gpu_id, paths):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    processor = LatentProcessor()
    processor.models = [processor.models[0]]

    success, failed, skipped = 0, 0, 0
    for wav_path in tqdm(paths, desc=f"GPU {gpu_id}"):
        status, result = processor.process_file(wav_path, gpu_id=0)
        if status == "success":
            success += 1
        elif status == "skipped":
            skipped += 1
        elif status == "failed":
            failed += 1
            print(f"❌ [GPU {gpu_id}] Failed: {wav_path} | See error_log.jsonl for details.")

    print(f"\n📊 [GPU {gpu_id}] Summary: ✅ {success} | ❌ {failed} | ⏭️ {skipped} out of {len(paths)} files\n")


class LatentProcessor:
    def __init__(self):
        self.completed_files = set()
        self.error_stats = defaultdict(int)
        self.models = self._load_models()

    def _load_models(self):
        print("🔄 Loading models for visible GPUs...")
        models = []
        for gpu_id in range(torch.cuda.device_count()):
            torch.cuda.set_device(gpu_id)
            try:
                pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
                pipeline.load_checkpoint(CHECKPOINT_DIR)
                model = pipeline.music_dcae.eval().to(gpu_id)
                models.append(model)
                print(f"✅ Model loaded on GPU {gpu_id} ({torch.cuda.get_device_name(gpu_id)})")
            except Exception as e:
                print(f"❌ Failed to load model on GPU {gpu_id}: {str(e)}")
                raise
        return models

    def _validate_audio(self, waveform, original_duration=None):
        if not torch.isfinite(waveform).all():
            raise ValueError("Audio contains NaN/inf values")
        if waveform.abs().max() < SILENCE_THRESHOLD:
            raise ValueError("Audio is silent or near-silent")
        if waveform.shape[-1] < MIN_SAMPLES:
            raise ValueError(f"Audio too short ({waveform.shape[-1]} < {MIN_SAMPLES} samples)")
        if waveform.shape[-1] > MAX_SAMPLES:
            raise ValueError(f"Audio too long ({waveform.shape[-1]} > {MAX_SAMPLES} samples)")

    def _validate_latents(self, latents, original_duration=None):
        if not torch.isfinite(latents).all():
            raise ValueError("Latents contain NaN/inf values")
        if latents.shape[:2] != LATENT_SHAPE:
            raise ValueError(f"Latent shape {latents.shape[:2]} != {LATENT_SHAPE}")
        std, mean = torch.std_mean(latents)
        if not (STD_RANGE[0] < std < STD_RANGE[1]):
            raise ValueError(f"Std dev {std:.4f} outside acceptable range {STD_RANGE}")
        if not (MEAN_RANGE[0] < mean < MEAN_RANGE[1]):
            raise ValueError(f"Mean {mean:.4f} outside acceptable range {MEAN_RANGE}")

    def _get_session_output_path(self, wav_path):
        path_obj = Path(wav_path)
        path_parts = path_obj.parts
        try:
            base_idx = path_parts.index("gcs-bucket")
            session_parts = path_parts[base_idx + 1:-1]
            session_dir = SAVE_DIR.joinpath(*session_parts)
            session_dir.mkdir(parents=True, exist_ok=True)
            return session_dir / f"{path_obj.stem}.pt"
        except ValueError:
            return SAVE_DIR / f"{path_obj.stem}.pt"

    def process_file(self, wav_path, gpu_id):
        self.current_file = wav_path
        if exclude_pattern.search(str(wav_path)):
            return "skipped", "drum file"
        try:
            if str(wav_path) in self.completed_files:
                return "skipped", "already processed"

            waveform, sr = torchaudio.load(wav_path)
            original_duration = waveform.shape[-1] / sr
            self._validate_audio(waveform)

            if waveform.shape[0] == 1:
                waveform = waveform.repeat(2, 1)
            waveform = waveform / (waveform.abs().max() + 1e-8)

            model = self.models[gpu_id]
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                device = torch.device(f"cuda:{gpu_id}")
                waveform = waveform.to(device)
                audio_batch = waveform.unsqueeze(0).float()
                audio_lengths = torch.tensor([waveform.shape[-1]], device=device)
                latents, latent_lengths = model.encode(
                    audios=audio_batch,
                    audio_lengths=audio_lengths,
                    sr=sr
                )

            latents = latents.float().squeeze(0).cpu()
            self._validate_latents(latents)

            out_path = self._get_session_output_path(wav_path)
            if out_path.exists():
                return "skipped", "already saved"

            latent_duration_sec = latents.shape[-1] * (DOWNSAMPLE_FACTOR / 44100.0)
            torch.save({
                'latents': latents,
                'length': latents.shape[-1],
                'original_path': str(wav_path),
                'original_duration': original_duration,
                'latent_duration': latent_duration_sec
            }, out_path)

            with open(PROGRESS_LOG, "a") as f:
                f.write(str(wav_path) + "\n")
            return "success", out_path

        # --- CHANGE: This block now logs all errors to the specified file ---
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)

            # 1. Create a dictionary for the log entry
            log_entry = {
                "file_path": str(wav_path),
                "error_type": error_type,
                "error_message": error_message,
            }

            # 2. Append the JSON entry to the log file
            try:
                with open(ERROR_LOG, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")
            except Exception as log_e:
                print(f"CRITICAL: Could not write to error log file: {log_e}")


            # 3. Update in-memory stats
            self.error_stats[error_type] += 1

            # 4. Return the correct status for console feedback
            if isinstance(e, ValueError):
                return "skipped", error_message # For validation errors, just skip
            else:
                return "failed", error_message # For other errors, mark as failed

        finally:
            torch.cuda.empty_cache()
            gc.collect()


def main():
    # Load full audio list
    with open(AUDIO_LIST_FILE, 'r') as f:
        audio_paths = [line.strip() for line in f if line.strip().endswith(".wav")]

    # Filter out completed files
    completed = set()
    if PROGRESS_LOG.exists():
        with open(PROGRESS_LOG, 'r') as f:
            completed = set(line.strip() for line in f if line.strip())

    remaining_paths = [p for p in audio_paths if p not in completed]
    print(f"📌 Found {len(completed)} previously processed files")
    print(f"🚀 Processing {len(remaining_paths)} remaining files across {NUM_GPUS} GPUs...")

    # Split workload across GPUs
    chunks = [remaining_paths[i::NUM_GPUS] for i in range(NUM_GPUS)]

    # Start GPU worker processes
    workers = []
    for gpu_id in range(torch.cuda.device_count()):
        proc = Process(target=gpu_worker, args=(gpu_id, chunks[gpu_id]))
        proc.start()
        workers.append(proc)

    for proc in workers:
        proc.join()

    print("\n✅ All GPU workers finished.")


if __name__ == "__main__":
    main()