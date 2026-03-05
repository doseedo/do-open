#!/usr/bin/env python3
"""
Batch ACE-Step v1.5 Latent Processor

Reads format_manifest.json to find audio files needing latents (has_latent: false),
processes them using DCAE and converts to v1.5 format, uploads to GCS bucket.

v1.5 Format:
  - Shape: [time, 64] (time-first, 64 channels)
  - Rate: 25 Hz
  - Converted from DCAE [8, 16, time] at ~10.8 Hz

Usage:
  python batch_latent_process_v15.py --gpus 0,1,2,3

Directory Structure:
  Audio source:  /home/arlo/gcs-bucket/{protools|protoolsA}/.../*.wav
  Latent dest:   /home/arlo/gcs-bucket/Latents2/{protools|protoolsA}/.../*.v15.pt
  Temp storage:  /mnt/models/batch_latents_v15/
  Manifest:      /home/arlo/gcs-bucket/Manifests/format_manifest.json
"""

import os
import sys
import json
import time
import subprocess
import gc
import shutil
from pathlib import Path
from typing import Dict, Tuple, Set
from dataclasses import dataclass
from multiprocessing import Process, Queue, Manager
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6"

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

# GPU memory constraints
L4_MAX_MEMORY_GB = 20

# ===================== CONFIGURATION =====================
@dataclass
class Config:
    sample_rate: int = 44100
    dcae_hop: int = 4096  # Old DCAE hop
    # Old DCAE output: [8, 16, T] at ~10.8 Hz
    dcae_latent_shape: Tuple[int, int] = (8, 16)
    dcae_latent_hz: float = 10.77  # 44100 / 512 / 8
    # New v1.5 format: [T, 64] at 25 Hz
    v15_latent_dim: int = 64
    v15_latent_hz: float = 25.0
    min_duration_sec: float = 0.5
    silence_threshold: float = 1e-6


# Paths
GCS_BUCKET = Path("/home/arlo/gcs-bucket")
PATHS_FILE = GCS_BUCKET / "Manifests" / "audio_paths_v2.3.txt"
LATENTS_DIR = GCS_BUCKET / "Latents2"  # New v1.5 latents folder
TEMP_DIR = Path("/mnt/models/batch_latents_v15")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
BATCH_SIZE = 1000
PROGRESS_FILE = TEMP_DIR / "progress.txt"  # Track completed paths


# ===================== AUDIO LOADING =====================

def load_audio(audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load and resample audio"""
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[-1] == 0:
        raise ValueError("empty_audio")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


# ===================== DCAE TO V1.5 CONVERSION =====================

def convert_dcae_to_v15(dcae_latent: torch.Tensor, config: Config) -> torch.Tensor:
    """
    Convert DCAE latent [8, 16, T] to v1.5 format [T', 64]

    Steps:
    1. Reshape [8, 16, T] -> [128, T]
    2. Average pairs to get [64, T]
    3. Interpolate from ~10.8 Hz to 25 Hz
    4. Transpose to [T', 64]
    """
    # Input: [8, 16, T]
    groups, features, time_old = dcae_latent.shape

    # Step 1: Reshape to [128, T]
    latent_flat = dcae_latent.reshape(groups * features, time_old)  # [128, T]

    # Step 2: Average pairs to get [64, T]
    # Combine every 2 channels: [128, T] -> [64, T]
    latent_64 = latent_flat.reshape(config.v15_latent_dim, 2, time_old).mean(dim=1)  # [64, T]

    # Step 3: Interpolate from 10.8 Hz to 25 Hz
    time_scale = config.v15_latent_hz / config.dcae_latent_hz
    time_new = int(time_old * time_scale)

    # Use linear interpolation
    latent_64 = latent_64.unsqueeze(0)  # [1, 64, T]
    latent_interp = F.interpolate(
        latent_64,
        size=time_new,
        mode='linear',
        align_corners=True
    ).squeeze(0)  # [64, T']

    # Step 4: Transpose to [T', 64] for v1.5 model format
    latent_v15 = latent_interp.permute(1, 0)  # [T', 64]

    return latent_v15


def extract_v15_latents(
    waveform: torch.Tensor,
    sr: int,
    dcae_model,
    device: torch.device,
    config: Config
) -> torch.Tensor:
    """Extract v1.5 format latents [T, 64] at 25 Hz"""
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Normalize
    waveform = waveform / (waveform.abs().max() + 1e-8)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        waveform = waveform.to(device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=device)

        # Get DCAE latents [1, 8, 16, T]
        dcae_latents, _ = dcae_model.encode(
            audios=audio_batch,
            audio_lengths=audio_lengths,
            sr=sr
        )

        # Convert to CPU for processing
        dcae_latents = dcae_latents.float().squeeze(0).cpu()  # [8, 16, T]

    # Validate DCAE output
    if not torch.isfinite(dcae_latents).all():
        raise ValueError("DCAE latents contain NaN/inf")
    if dcae_latents.shape[:2] != config.dcae_latent_shape:
        raise ValueError(f"DCAE shape {dcae_latents.shape[:2]} != {config.dcae_latent_shape}")

    # Convert to v1.5 format
    latent_v15 = convert_dcae_to_v15(dcae_latents, config)  # [T, 64]

    # Validate output
    if latent_v15.shape[1] != config.v15_latent_dim:
        raise ValueError(f"v1.5 latent dim {latent_v15.shape[1]} != {config.v15_latent_dim}")

    return latent_v15


def get_latent_path(audio_path: str, base_dir: Path) -> Path:
    """Get output path for v1.5 latent file mirroring audio structure.

    Takes full audio path like /home/arlo/gcs-bucket/protools/.../*.wav
    Outputs to base_dir with same structure: base_dir/protools/.../*.v15.pt
    """
    p = Path(audio_path)
    stem = p.stem

    # Extract relative path from gcs-bucket
    try:
        rel_path = p.relative_to(GCS_BUCKET)
    except ValueError:
        # If not under GCS_BUCKET, use just the parent dirs
        rel_path = p.parent.name / p.name

    return base_dir / rel_path.parent / f"{stem}.v15.pt"


# ===================== PATH LIST MANAGEMENT =====================

def load_audio_paths() -> list:
    """Load list of audio paths from text file"""
    with open(PATHS_FILE, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def load_completed_paths() -> Set[str]:
    """Load set of already-completed paths from progress file"""
    if not PROGRESS_FILE.exists():
        return set()
    with open(PROGRESS_FILE, 'r') as f:
        return set(line.strip() for line in f if line.strip())


def save_completed_paths(completed: Set[str]):
    """Save completed paths to progress file"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'w') as f:
        f.write('\n'.join(sorted(completed)))


def get_paths_needing_latents(all_paths: list, completed: Set[str]) -> list:
    """Filter to paths that haven't been processed yet.

    Also checks if latent file already exists in Latents2.
    """
    needs_processing = []
    for audio_path in all_paths:
        if audio_path in completed:
            continue
        # Check if latent already exists
        latent_path = get_latent_path(audio_path, LATENTS_DIR)
        if latent_path.exists():
            completed.add(audio_path)
            continue
        needs_processing.append(audio_path)
    return needs_processing


def append_to_progress(paths: Set[str]):
    """Append newly completed paths to progress file"""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, 'a') as f:
        for p in paths:
            f.write(p + '\n')


# ===================== BATCH UPLOAD =====================

def upload_batch_to_gcs(temp_dir: Path, dest_base: Path):
    if not any(temp_dir.iterdir()):
        return
    cmd = ["gsutil", "-m", "-q", "cp", "-r", str(temp_dir) + "/*", str(dest_base) + "/"]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[UPLOAD] Batch uploaded to {dest_base}")
    except subprocess.CalledProcessError as e:
        print(f"[UPLOAD ERROR] {e.stderr.decode()}")
        raise


def clear_temp_dir(temp_dir: Path):
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)


# ===================== GPU WORKER =====================

def gpu_worker(
    gpu_id: int,
    work_queue: Queue,
    result_queue: Queue,
    progress_queue: Queue,
    shutdown_event,
    config: Config,
    num_preload: int = 8
):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from queue import Queue as LocalQueue
    import threading

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = torch.device("cuda:0")

    torch.cuda.set_per_process_memory_fraction(L4_MAX_MEMORY_GB / 24.0, device=0)
    torch.cuda.empty_cache()

    # Load DCAE model
    print(f"[GPU {gpu_id}] Loading DCAE model...")
    sys.path.insert(0, '/home/arlo/Data/ACE-Step')
    from acestep.music_dcae.music_dcae_pipeline import MusicDCAE

    dcae_model = MusicDCAE(
        dcae_checkpoint_path=f'{CHECKPOINT_DIR}/music_dcae_f8c8',
        vocoder_checkpoint_path=f'{CHECKPOINT_DIR}/music_vocoder'
    ).eval().to(device)

    torch.cuda.empty_cache()
    gc.collect()
    print(f"[GPU {gpu_id}] DCAE loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    processed_count = 0

    def preload_audio(item):
        """Load audio in background thread"""
        if item is None:
            return None
        rel_path, audio_path = item
        try:
            waveform, sr = load_audio(Path(audio_path), config.sample_rate)
            duration = waveform.shape[-1] / sr

            skip_reason = None
            if duration < config.min_duration_sec:
                skip_reason = f"too_short:{duration:.1f}s"
            elif waveform.abs().max() < config.silence_threshold:
                skip_reason = "silent"
            elif not torch.isfinite(waveform).all():
                skip_reason = "nan_inf"

            return (rel_path, audio_path, waveform, sr, duration, skip_reason, None)
        except FileNotFoundError:
            return (rel_path, audio_path, None, None, None, "file_not_found", None)
        except ValueError as e:
            if str(e) == "empty_audio":
                return (rel_path, audio_path, None, None, None, "empty_audio", None)
            return (rel_path, audio_path, None, None, None, None, str(e))
        except Exception as e:
            return (rel_path, audio_path, None, None, None, None, str(e))

    preload_queue = LocalQueue(maxsize=num_preload)
    stop_preload = threading.Event()

    def preload_worker():
        from queue import Empty
        with ThreadPoolExecutor(max_workers=num_preload) as executor:
            futures = []
            got_poison_pill = False

            while not stop_preload.is_set():
                while len(futures) < num_preload * 2 and not got_poison_pill:
                    try:
                        item = work_queue.get(timeout=0.5)
                        if item is None:
                            got_poison_pill = True
                            break
                        futures.append(executor.submit(preload_audio, item))
                    except Empty:
                        break
                    except Exception:
                        break

                done = [f for f in futures if f.done()]
                for f in done:
                    futures.remove(f)
                    try:
                        result = f.result()
                        preload_queue.put(result, timeout=10)
                    except Exception:
                        pass

                if got_poison_pill and not futures:
                    break
                if not done and not got_poison_pill:
                    time.sleep(0.01)

            preload_queue.put(None)

    preload_thread = threading.Thread(target=preload_worker, daemon=True)
    preload_thread.start()

    consecutive_timeouts = 0
    max_consecutive_timeouts = 30

    while not shutdown_event.is_set():
        try:
            item = preload_queue.get(timeout=1)
            consecutive_timeouts = 0

            if item is None:
                print(f"[GPU {gpu_id}] Received end signal")
                break

            rel_path, audio_path, waveform, sr, duration, skip_reason, error = item

            if skip_reason:
                result_queue.put(("skipped", rel_path, skip_reason))
                progress_queue.put((gpu_id, processed_count))
                continue
            if error:
                result_queue.put(("failed", rel_path, error))
                progress_queue.put((gpu_id, processed_count))
                continue

            try:
                # Extract v1.5 format latents
                latents = extract_v15_latents(waveform, sr, dcae_model, device, config)

                out_path = get_latent_path(rel_path, TEMP_DIR)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Save in v1.5 format
                torch.save({
                    "latents": latents,  # [T, 64] at 25 Hz
                    "length": latents.shape[0],
                    "latent_hz": config.v15_latent_hz,
                    "latent_dim": config.v15_latent_dim,
                    "original_path": str(audio_path),
                    "original_duration": duration,
                    "format_version": "v1.5"
                }, out_path)

                result_queue.put(("success", rel_path))
                processed_count += 1

            except torch.cuda.OutOfMemoryError as e:
                print(f"[GPU {gpu_id}] CUDA OOM on {rel_path}: {e}")
                result_queue.put(("skipped", rel_path, "cuda_oom"))
                torch.cuda.empty_cache()
                gc.collect()

            except Exception as e:
                result_queue.put(("failed", rel_path, str(e)))

            finally:
                torch.cuda.empty_cache()
                progress_queue.put((gpu_id, processed_count))

        except Exception:
            consecutive_timeouts += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                if not preload_thread.is_alive():
                    break
                consecutive_timeouts = 0

    stop_preload.set()
    preload_thread.join(timeout=5)
    print(f"[GPU {gpu_id}] Finished, processed {processed_count}")


# ===================== MAIN =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch ACE-Step v1.5 latent processor")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Process but don't upload or update manifest")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0=all)")
    parser.add_argument("--preload-workers", type=int, default=8, help="Number of audio preload threads")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    batch_size = args.batch_size

    print(f"Loading paths from {PATHS_FILE}...")
    all_paths = load_audio_paths()
    print(f"Total audio files in list: {len(all_paths)}")

    completed = load_completed_paths()
    print(f"Already completed: {len(completed)}")

    entries = get_paths_needing_latents(all_paths, completed)
    total_needing = len(entries)

    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"Found {total_needing} files needing v1.5 latents, processing {len(entries)}")

    if not entries:
        print("Nothing to process!")
        return

    # Keep temp dir but clear old files
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    config = Config()

    manager = Manager()
    work_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    shutdown_event = manager.Event()

    workers = []
    for gpu_id in gpu_ids:
        p = Process(target=gpu_worker, args=(
            gpu_id, work_queue, result_queue, progress_queue,
            shutdown_event, config, args.preload_workers
        ))
        p.start()
        workers.append(p)

    # entries is list of audio paths
    for audio_path in entries:
        work_queue.put((audio_path, audio_path))

    for _ in workers:
        work_queue.put(None)

    from tqdm import tqdm
    pbar = tqdm(total=len(entries), desc="Processing v1.5")

    processed_in_batch = set()
    skipped_in_batch = {}
    total_processed = 0
    total_skipped = 0
    total_failed = 0

    total_to_process = len(entries)

    while total_processed + total_skipped + total_failed < total_to_process:
        try:
            result = result_queue.get(timeout=5)

            if result[0] == "success":
                _, rel_path = result
                processed_in_batch.add(rel_path)
                total_processed += 1
            elif result[0] == "skipped":
                _, rel_path, reason = result
                skipped_in_batch[rel_path] = reason
                total_skipped += 1
            else:
                _, rel_path, error = result
                total_failed += 1

            pbar.update(1)
            pbar.set_postfix({
                "ok": total_processed,
                "skip": total_skipped,
                "fail": total_failed,
                "batch": len(processed_in_batch)
            })

            if len(processed_in_batch) + len(skipped_in_batch) >= batch_size:
                pbar.set_description("Uploading...")

                if not args.dry_run:
                    upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)
                    # Track all completed (both successful and skipped)
                    all_done = processed_in_batch | set(skipped_in_batch.keys())
                    append_to_progress(all_done)
                    print(f"\n[BATCH] Uploaded {len(processed_in_batch)}, skipped {len(skipped_in_batch)}")

                processed_in_batch.clear()
                skipped_in_batch = {}
                clear_temp_dir(TEMP_DIR)
                pbar.set_description("Processing v1.5")

        except Exception:
            if not any(p.is_alive() for p in workers):
                break

    pbar.close()

    # Final batch
    if (processed_in_batch or skipped_in_batch) and not args.dry_run:
        print(f"[FINAL] Uploading {len(processed_in_batch)} files...")
        upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)
        all_done = processed_in_batch | set(skipped_in_batch.keys())
        append_to_progress(all_done)

    shutdown_event.set()
    for p in workers:
        p.join(timeout=10)

    clear_temp_dir(TEMP_DIR)

    print("\n" + "=" * 50)
    print("SUMMARY (v1.5 Latents)")
    print("=" * 50)
    print(f"  Processed: {total_processed}")
    print(f"  Skipped:   {total_skipped}")
    print(f"  Failed:    {total_failed}")
    total = total_processed + total_skipped + total_failed
    if total > 0:
        print(f"  Success rate: {100*total_processed/total:.1f}%")
    print(f"  Output format: [T, 64] at 25 Hz")
    print(f"  Output dir: {LATENTS_DIR}")


if __name__ == "__main__":
    main()
