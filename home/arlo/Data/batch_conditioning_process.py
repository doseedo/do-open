#!/usr/bin/env python3
"""
Batch Conditioning Processor

Reads format_manifest.json to find audio files needing conditioning features,
processes them in batches, uploads to GCS bucket, and updates manifest periodically.

Usage:
  python batch_conditioning_process.py --gpus 0,1,2,3

Directory Structure:
  Audio source:  /home/arlo/gcs-bucket/{protools|protoolsA}/.../*.wav
  Latent dest:   /home/arlo/gcs-bucket/Latents/{protools|protoolsA}/.../*.{amp,rframe,rbend,f0,f0_masked,onsets}.npy
  Temp storage:  /mnt/models/batch_conditioning/
  Manifest:      /home/arlo/gcs-bucket/Manifests/format_manifest.json
"""

import os
import sys
import json
import time
import signal
import subprocess
import gc
import traceback
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from dataclasses import dataclass, field
from multiprocessing import Process, Queue, Manager
from collections import defaultdict
from datetime import datetime
import threading

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"

import numpy as np
import torch
import torchaudio

# ===================== CONFIGURATION =====================
@dataclass
class Config:
    sample_rate: int = 44100
    hop_length: int = 4096
    n_fft: int = 8192
    fmin: float = 65.41   # C2
    fmax: float = 2093.00  # C7

    min_duration_sec: float = 0.5
    max_duration_sec: float = 480.0
    silence_threshold: float = 1e-6

    @property
    def frame_rate(self) -> float:
        return self.sample_rate / self.hop_length


# Paths
GCS_BUCKET = Path("/home/arlo/gcs-bucket")
MANIFEST_PATH = GCS_BUCKET / "Manifests" / "format_manifest.json"
LATENTS_DIR = GCS_BUCKET / "Latents"
TEMP_DIR = Path("/mnt/models/batch_conditioning")
BATCH_SIZE = 1000  # Upload and update manifest every N files


# ===================== CONDITIONING EXTRACTION =====================

def load_audio(audio_path: Path, target_sr: int = 44100) -> Tuple[torch.Tensor, int]:
    """Load and resample audio to target sample rate"""
    waveform, sr = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        sr = target_sr
    return waveform, sr


def extract_conditioning_features(
    waveform: torch.Tensor,
    sr: int,
    config: Config,
    device: str = "cuda"
) -> Dict[str, np.ndarray]:
    """Extract conditioning features: amp, rframe, rbend, f0, f0_masked, onsets"""
    import torchcrepe

    y = waveform[0]  # Mono

    win = config.n_fft
    hop = config.hop_length
    if y.numel() < win:
        y = torch.nn.functional.pad(y, (0, win - y.numel()))

    # Resample to 16kHz for torchcrepe
    crepe_sr = 16000
    crepe_hop = int(crepe_sr / config.frame_rate)
    y_16k = torchaudio.functional.resample(y.unsqueeze(0), sr, crepe_sr).squeeze(0)

    # RMS amplitude
    frames = y.unfold(0, win, hop)
    amp = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)
    if amp.max() > 0:
        amp = amp / amp.max()
    amp = amp.numpy().astype(np.float32)

    # F0 with torchcrepe
    y_crepe = y_16k.unsqueeze(0).to(device)

    with torch.inference_mode():
        f0, periodicity = torchcrepe.predict(
            y_crepe,
            sample_rate=crepe_sr,
            hop_length=crepe_hop,
            fmin=config.fmin,
            fmax=config.fmax,
            pad=True,
            model="tiny",
            batch_size=256,
            device=device,
            return_periodicity=True
        )
        f0 = f0[0].cpu()
        periodicity = periodicity[0].cpu()

    # Align lengths
    T = min(len(amp), len(f0))
    amp = amp[:T]
    f0 = f0[:T]
    periodicity = periodicity[:T]

    # Voiced mask (rframe)
    vmask = ((periodicity > 0.5) & (torch.from_numpy(amp) > 0.02)).float().numpy()

    # Pitch bend (rbend)
    f0_safe = np.where(f0.numpy() > 0, f0.numpy(), 1.0)
    rbend = 12.0 * np.log2(f0_safe / 440.0)
    rbend = np.where(np.isfinite(rbend), rbend, 0.0).astype(np.float32)
    rbend = rbend * vmask

    # F0 and masked F0
    f0_np = f0.numpy().astype(np.float32)
    f0_masked = f0_np * vmask

    # Onsets
    amp_diff = np.diff(amp, prepend=amp[0])
    onsets = (amp_diff > 0.05).astype(np.float32)

    return {
        "amp": amp.astype(np.float32),
        "rframe": vmask.astype(np.float32),
        "rbend": rbend.astype(np.float32),
        "f0": f0_np,
        "f0_masked": f0_masked.astype(np.float32),
        "onsets": onsets.astype(np.float32)
    }


def get_conditioning_paths(rel_path: str, base_dir: Path) -> Dict[str, Path]:
    """Get output paths for conditioning features mirroring audio structure"""
    # rel_path is like: protools/2025-03-28/New/.../file.wav
    p = Path(rel_path)
    stem = p.stem
    parent = p.parent  # protools/2025-03-28/New/.../

    out_dir = base_dir / parent

    return {
        "amp": out_dir / f"{stem}.amp.npy",
        "rframe": out_dir / f"{stem}.rframe.npy",
        "rbend": out_dir / f"{stem}.rbend.npy",
        "f0": out_dir / f"{stem}.f0.npy",
        "f0_masked": out_dir / f"{stem}.f0_masked.npy",
        "onsets": out_dir / f"{stem}.onsets.npy"
    }


# ===================== MANIFEST MANAGEMENT =====================

def load_manifest() -> Dict:
    """Load manifest from disk"""
    with open(MANIFEST_PATH, 'r') as f:
        return json.load(f)


def save_manifest(manifest: Dict):
    """Save manifest to disk"""
    manifest["generated_at"] = time.time()
    manifest["generated_at_iso"] = datetime.now().isoformat()

    # Write to temp file first, then atomic rename
    temp_path = MANIFEST_PATH.with_suffix('.json.tmp')
    with open(temp_path, 'w') as f:
        json.dump(manifest, f)
    temp_path.rename(MANIFEST_PATH)


def get_entries_needing_conditioning(manifest: Dict) -> List[Dict]:
    """Filter manifest entries that need conditioning"""
    return [e for e in manifest["entries"] if not e.get("has_conditioning", False)]


def update_manifest_entries(manifest: Dict, processed_paths: Set[str]):
    """Update manifest entries to mark conditioning as complete"""
    for entry in manifest["entries"]:
        if entry["path"] in processed_paths:
            entry["has_conditioning"] = True

    # Update stats
    with_cond = sum(1 for e in manifest["entries"] if e.get("has_conditioning", False))
    manifest["stats"]["with_conditioning"] = with_cond
    manifest["stats"]["needs_conditioning"] = manifest["stats"]["total_audio"] - with_cond
    manifest["stats"]["pct_conditioning"] = 100.0 * with_cond / manifest["stats"]["total_audio"]


# ===================== BATCH UPLOAD =====================

def upload_batch_to_gcs(temp_dir: Path, dest_base: Path):
    """Upload processed files from temp dir to GCS bucket using gsutil"""
    if not any(temp_dir.iterdir()):
        return

    # Use gsutil -m for parallel uploads
    # Copy entire temp directory structure to destination
    cmd = [
        "gsutil", "-m", "-q", "cp", "-r",
        str(temp_dir) + "/*",
        str(dest_base) + "/"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[UPLOAD] Uploaded batch to {dest_base}")
    except subprocess.CalledProcessError as e:
        print(f"[UPLOAD ERROR] gsutil failed: {e.stderr.decode()}")
        raise


def clear_temp_dir(temp_dir: Path):
    """Clear temporary directory"""
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
    config: Config
):
    """Worker process for a single GPU"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = "cuda:0"

    # Pre-import torchcrepe to load model once
    import torchcrepe

    processed_count = 0

    while not shutdown_event.is_set():
        try:
            item = work_queue.get(timeout=1)
            if item is None:  # Poison pill
                break

            rel_path, audio_path = item

            try:
                # Load audio
                waveform, sr = load_audio(Path(audio_path), config.sample_rate)

                # Validate
                duration = waveform.shape[-1] / sr
                if duration < config.min_duration_sec or duration > config.max_duration_sec:
                    raise ValueError(f"Duration {duration:.1f}s out of range")
                if waveform.abs().max() < config.silence_threshold:
                    raise ValueError("Silent audio")
                if not torch.isfinite(waveform).all():
                    raise ValueError("Contains NaN/inf")

                # Extract features
                features = extract_conditioning_features(waveform, sr, config, device)

                # Save to temp directory (mirroring structure)
                out_paths = get_conditioning_paths(rel_path, TEMP_DIR)
                out_paths["amp"].parent.mkdir(parents=True, exist_ok=True)

                for feat_name, feat_data in features.items():
                    np.save(out_paths[feat_name], feat_data)

                result_queue.put(("success", rel_path))
                processed_count += 1

            except Exception as e:
                result_queue.put(("failed", rel_path, str(e)))

            finally:
                torch.cuda.empty_cache()
                progress_queue.put((gpu_id, processed_count))

        except Exception:
            continue

    print(f"[GPU {gpu_id}] Worker finished, processed {processed_count} files")


# ===================== MAIN =====================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch conditioning processor")
    parser.add_argument("--gpus", default="0", help="Comma-separated GPU IDs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Upload batch size (default: {BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true", help="Don't upload or update manifest")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    batch_size = args.batch_size

    print(f"Loading manifest from {MANIFEST_PATH}...")
    manifest = load_manifest()

    entries = get_entries_needing_conditioning(manifest)
    print(f"Found {len(entries)} files needing conditioning")

    if not entries:
        print("Nothing to process!")
        return

    # Setup temp directory
    clear_temp_dir(TEMP_DIR)

    config = Config()

    # Create queues and events
    manager = Manager()
    work_queue = Queue()
    result_queue = Queue()
    progress_queue = Queue()
    shutdown_event = manager.Event()

    # Start workers
    workers = []
    for gpu_id in gpu_ids:
        p = Process(target=gpu_worker, args=(
            gpu_id, work_queue, result_queue, progress_queue,
            shutdown_event, config
        ))
        p.start()
        workers.append(p)

    # Fill work queue
    for entry in entries:
        rel_path = entry["path"]
        audio_path = GCS_BUCKET / rel_path
        work_queue.put((rel_path, str(audio_path)))

    # Add poison pills
    for _ in workers:
        work_queue.put(None)

    # Process results
    from tqdm import tqdm
    pbar = tqdm(total=len(entries), desc="Processing")

    processed_in_batch = set()
    total_processed = 0
    total_failed = 0

    while total_processed + total_failed < len(entries):
        try:
            result = result_queue.get(timeout=5)

            if result[0] == "success":
                _, rel_path = result
                processed_in_batch.add(rel_path)
                total_processed += 1
            else:
                _, rel_path, error = result
                total_failed += 1
                # Optionally log errors

            pbar.update(1)
            pbar.set_postfix({
                "ok": total_processed,
                "fail": total_failed,
                "batch": len(processed_in_batch)
            })

            # Batch upload when threshold reached
            if len(processed_in_batch) >= batch_size:
                pbar.set_description("Uploading batch...")

                if not args.dry_run:
                    # Upload to GCS
                    upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)

                    # Update manifest
                    update_manifest_entries(manifest, processed_in_batch)
                    save_manifest(manifest)
                    print(f"\n[BATCH] Uploaded {len(processed_in_batch)} files, updated manifest")

                # Clear for next batch
                processed_in_batch.clear()
                clear_temp_dir(TEMP_DIR)
                pbar.set_description("Processing")

        except Exception:
            # Check workers still alive
            if not any(p.is_alive() for p in workers):
                break

    pbar.close()

    # Final batch upload
    if processed_in_batch and not args.dry_run:
        print(f"[FINAL] Uploading final batch of {len(processed_in_batch)} files...")
        upload_batch_to_gcs(TEMP_DIR, LATENTS_DIR)
        update_manifest_entries(manifest, processed_in_batch)
        save_manifest(manifest)

    # Cleanup
    shutdown_event.set()
    for p in workers:
        p.join(timeout=10)

    clear_temp_dir(TEMP_DIR)

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"  Processed: {total_processed}")
    print(f"  Failed: {total_failed}")
    print(f"  Success rate: {100*total_processed/(total_processed+total_failed):.1f}%")


if __name__ == "__main__":
    main()
