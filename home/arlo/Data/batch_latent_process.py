#!/usr/bin/env python3
"""
Batch DCAE Latent Processor

Reads format_manifest.json to find audio files needing latents (has_latent: false),
processes them in batches, uploads to GCS bucket, and updates manifest periodically.

Usage:
  python batch_latent_process.py --gpus 0,1,2,3

Directory Structure:
  Audio source:  /home/arlo/gcs-bucket/{protools|protoolsA}/.../*.wav
  Latent dest:   /home/arlo/gcs-bucket/Latents/{protools|protoolsA}/.../*.dcae.pt
  Temp storage:  /mnt/models/batch_latents/
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
# L4 GPU (24GB) memory constraints
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:256,garbage_collection_threshold:0.6"

import numpy as np
import torch
import torchaudio

# L4 memory limit: reserve ~4GB for system/overhead, use ~20GB max
L4_MAX_MEMORY_GB = 20
# Note: No duration/file size limits - let OOM handler catch memory issues at runtime

# ===================== CONFIGURATION =====================
@dataclass
class Config:
    sample_rate: int = 44100
    dcae_hop: int = 4096
    latent_shape: Tuple[int, int] = (8, 16)
    min_duration_sec: float = 0.5
    silence_threshold: float = 1e-6


# Paths
GCS_BUCKET = Path("/home/arlo/gcs-bucket")
MANIFEST_PATH = GCS_BUCKET / "Manifests" / "format_manifest.json"
LATENTS_DIR = GCS_BUCKET / "Latents"
TEMP_DIR = Path("/mnt/models/batch_latents")
CHECKPOINT_DIR = "/home/arlo/Data/ACE-Step/checkpoints"
BATCH_SIZE = 1000


# ===================== DCAE EXTRACTION =====================

def load_audio(audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load and resample audio"""
    waveform, sr = torchaudio.load(str(audio_path))
    # Check for empty audio (valid header but 0 samples)
    if waveform.shape[-1] == 0:
        raise ValueError("empty_audio")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def extract_dcae_latents(
    waveform: torch.Tensor,
    sr: int,
    dcae_model,
    device: torch.device,
    config: Config
) -> torch.Tensor:
    """Extract DCAE latents [8, 16, T]"""
    # Ensure stereo
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(2, 1)

    # Normalize
    waveform = waveform / (waveform.abs().max() + 1e-8)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
        waveform = waveform.to(device)
        audio_batch = waveform.unsqueeze(0).float()
        audio_lengths = torch.tensor([waveform.shape[-1]], device=device)

        latents, _ = dcae_model.encode(
            audios=audio_batch,
            audio_lengths=audio_lengths,
            sr=sr
        )

    latents = latents.float().squeeze(0).cpu()

    if not torch.isfinite(latents).all():
        raise ValueError("Latents contain NaN/inf")
    if latents.shape[:2] != config.latent_shape:
        raise ValueError(f"Shape {latents.shape[:2]} != {config.latent_shape}")

    return latents


def get_latent_path(rel_path: str, base_dir: Path) -> Path:
    """Get output path for latent file mirroring audio structure"""
    p = Path(rel_path)
    stem = p.stem
    parent = p.parent
    return base_dir / parent / f"{stem}.dcae.pt"


# ===================== MANIFEST MANAGEMENT =====================

def load_manifest() -> Dict:
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)


def save_manifest(manifest: Dict):
    manifest["generated_at"] = time.time()
    manifest["generated_at_iso"] = datetime.now().isoformat()
    temp_path = MANIFEST_PATH.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(manifest, f)
    temp_path.rename(MANIFEST_PATH)


def get_entries_needing_latents(manifest: Dict) -> list:
    """Get entries where has_latent is False (not True, not 'skipped')"""
    return [
        e for e in manifest["entries"]
        if e.get("has_latent") is False  # Explicit False, not True or "skipped"
    ]


def update_manifest_entries(manifest: Dict, processed_paths: Set[str], skipped_paths: Dict[str, str]):
    """Update manifest with processed (True) and skipped ('skipped') entries

    Args:
        processed_paths: Set of paths that were successfully processed
        skipped_paths: Dict mapping path -> skip_reason
    """
    for entry in manifest["entries"]:
        if entry["path"] in processed_paths:
            entry["has_latent"] = True
        elif entry["path"] in skipped_paths:
            entry["has_latent"] = "skipped"
            entry["skip_reason"] = skipped_paths[entry["path"]]

    # Update stats
    with_latent = sum(1 for e in manifest["entries"] if e.get("has_latent") is True)
    skipped = sum(1 for e in manifest["entries"] if e.get("has_latent") == "skipped")
    needs = sum(1 for e in manifest["entries"] if e.get("has_latent") is False)

    manifest["stats"]["with_latent"] = with_latent
    manifest["stats"]["skipped_latent"] = skipped
    manifest["stats"]["needs_latent"] = needs
    manifest["stats"]["pct_latent"] = 100.0 * with_latent / manifest["stats"]["total_audio"]


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

    # L4 GPU memory constraints (24GB total, use max 20GB)
    torch.cuda.set_per_process_memory_fraction(L4_MAX_MEMORY_GB / 24.0, device=0)
    torch.cuda.empty_cache()

    # Load DCAE model
    print(f"[GPU {gpu_id}] Loading DCAE model (L4 memory limit: {L4_MAX_MEMORY_GB}GB)...")
    from acestep.pipeline_ace_step import ACEStepPipeline
    pipeline = ACEStepPipeline(checkpoint_dir=CHECKPOINT_DIR)
    pipeline.load_checkpoint(CHECKPOINT_DIR)
    dcae_model = pipeline.music_dcae.eval().to(device)
    del pipeline  # Free pipeline memory, keep only dcae
    torch.cuda.empty_cache()
    gc.collect()
    print(f"[GPU {gpu_id}] DCAE model loaded, VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")

    processed_count = 0
    print(f"[GPU {gpu_id}] Using {num_preload} preload workers")

    def preload_audio(item):
        """Load audio in background thread"""
        if item is None:
            return None
        rel_path, audio_path = item
        try:
            waveform, sr = load_audio(Path(audio_path), config.sample_rate)
            duration = waveform.shape[-1] / sr

            # Check skip conditions (only skip truly unusable audio, let OOM handle memory limits)
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

    # Preload queue for ready-to-process items
    preload_queue = LocalQueue(maxsize=num_preload)
    stop_preload = threading.Event()

    def preload_worker():
        """Background thread to preload audio"""
        from queue import Empty
        items_submitted = 0
        items_completed = 0

        with ThreadPoolExecutor(max_workers=num_preload) as executor:
            futures = []
            got_poison_pill = False

            while not stop_preload.is_set():
                # Submit new preload tasks (keep queue full)
                while len(futures) < num_preload * 2 and not got_poison_pill:
                    try:
                        item = work_queue.get(timeout=0.5)
                        if item is None:
                            got_poison_pill = True
                            break
                        futures.append(executor.submit(preload_audio, item))
                        items_submitted += 1
                    except Empty:
                        break  # Queue temporarily empty, continue processing
                    except Exception as e:
                        print(f"[PRELOAD] Error getting from queue: {e}")
                        break

                # Collect completed futures
                done = [f for f in futures if f.done()]
                for f in done:
                    futures.remove(f)
                    try:
                        result = f.result()
                        preload_queue.put(result, timeout=10)
                        items_completed += 1
                    except Exception as e:
                        print(f"[PRELOAD] Error processing result: {e}")

                # Only exit when all work is done
                if got_poison_pill and not futures:
                    break

                # Small sleep to prevent busy loop
                if not done and not got_poison_pill:
                    time.sleep(0.01)

            # Signal end to consumer
            preload_queue.put(None)
            print(f"[PRELOAD] Thread done: submitted={items_submitted}, completed={items_completed}")

    # Start preload thread
    preload_thread = threading.Thread(target=preload_worker, daemon=True)
    preload_thread.start()

    consecutive_timeouts = 0
    max_consecutive_timeouts = 30  # 30 seconds of no work = something wrong

    while not shutdown_event.is_set():
        try:
            item = preload_queue.get(timeout=1)
            consecutive_timeouts = 0  # Reset on successful get

            if item is None:
                print(f"[GPU {gpu_id}] Received end signal from preload thread")
                break

            rel_path, audio_path, waveform, sr, duration, skip_reason, error = item

            # Handle preload errors
            if skip_reason:
                result_queue.put(("skipped", rel_path, skip_reason))
                progress_queue.put((gpu_id, processed_count))
                continue
            if error:
                result_queue.put(("failed", rel_path, error))
                progress_queue.put((gpu_id, processed_count))
                continue

            try:
                latents = extract_dcae_latents(waveform, sr, dcae_model, device, config)

                out_path = get_latent_path(rel_path, TEMP_DIR)
                out_path.parent.mkdir(parents=True, exist_ok=True)

                torch.save({
                    "latents": latents,
                    "length": latents.shape[-1],
                    "original_path": str(audio_path),
                    "original_duration": duration
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

        except Exception as e:
            consecutive_timeouts += 1
            if consecutive_timeouts >= max_consecutive_timeouts:
                print(f"[GPU {gpu_id}] No work for {max_consecutive_timeouts}s, checking preload thread...")
                if not preload_thread.is_alive():
                    print(f"[GPU {gpu_id}] Preload thread died!")
                    break
                consecutive_timeouts = 0  # Reset and keep waiting

    stop_preload.set()
    preload_thread.join(timeout=5)
    print(f"[GPU {gpu_id}] Finished, processed {processed_count}")


# ===================== MAIN =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch DCAE latent processor")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dry-run", action="store_true", help="Process but don't upload or update manifest")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of files to process (0=all)")
    parser.add_argument("--preload-workers", type=int, default=8, help="Number of audio preload threads (default: 8)")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    batch_size = args.batch_size

    print(f"Loading manifest from {MANIFEST_PATH}...")
    manifest = load_manifest()

    entries = get_entries_needing_latents(manifest)
    total_needing = len(entries)

    if args.limit > 0:
        entries = entries[:args.limit]

    print(f"Found {total_needing} files needing latents, processing {len(entries)}")

    if not entries:
        print("Nothing to process!")
        return

    clear_temp_dir(TEMP_DIR)
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

    for entry in entries:
        rel_path = entry["path"]
        audio_path = GCS_BUCKET / rel_path
        work_queue.put((rel_path, str(audio_path)))

    for _ in workers:
        work_queue.put(None)

    from tqdm import tqdm
    pbar = tqdm(total=len(entries), desc="Processing")

    processed_in_batch = set()
    skipped_in_batch = {}  # path -> reason
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
            else:  # failed
                _, rel_path, error = result
                total_failed += 1

            pbar.update(1)
            pbar.set_postfix({
                "ok": total_processed,
                "skip": total_skipped,
                "fail": total_failed,
                "batch": len(processed_in_batch)
            })

            # Batch upload when threshold reached (count both processed and skipped)
            if len(processed_in_batch) + len(skipped_in_batch) >= batch_size:
                pbar.set_description("Uploading...")

                if not args.dry_run:
                    upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)
                    update_manifest_entries(manifest, processed_in_batch, skipped_in_batch)
                    save_manifest(manifest)
                    print(f"\n[BATCH] Uploaded {len(processed_in_batch)}, skipped {len(skipped_in_batch)}, manifest updated")

                processed_in_batch.clear()
                skipped_in_batch = {}
                clear_temp_dir(TEMP_DIR)
                pbar.set_description("Processing")

        except Exception:
            if not any(p.is_alive() for p in workers):
                break

    pbar.close()

    # Final batch
    if (processed_in_batch or skipped_in_batch) and not args.dry_run:
        print(f"[FINAL] Uploading {len(processed_in_batch)} files, marking {len(skipped_in_batch)} skipped...")
        upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)
        update_manifest_entries(manifest, processed_in_batch, skipped_in_batch)
        save_manifest(manifest)

    shutdown_event.set()
    for p in workers:
        p.join(timeout=10)

    clear_temp_dir(TEMP_DIR)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Processed: {total_processed}")
    print(f"  Skipped:   {total_skipped} (marked has_latent: 'skipped')")
    print(f"  Failed:    {total_failed} (will retry on next run)")
    total = total_processed + total_skipped + total_failed
    if total > 0:
        print(f"  Success rate: {100*total_processed/total:.1f}%")


if __name__ == "__main__":
    main()
