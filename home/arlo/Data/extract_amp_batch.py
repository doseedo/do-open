#!/usr/bin/env python3
"""
Batch Amplitude Envelope Extractor

Extracts .amp.npy files for all audio entries that don't have conditioning yet.
Uses format_manifest.json to skip entries that already have conditioning.

Amplitude extraction is pure math (RMS envelope) — no GPU, no neural network.
Runs with ThreadPoolExecutor for I/O parallelism + batch saving.

Usage:
    python3 extract_amp_batch.py --workers 32
    python3 extract_amp_batch.py --workers 32 --limit 1000  # test run
    python3 extract_amp_batch.py --workers 32 --dry-run     # count only
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Queue
from threading import Lock, Thread

import numpy as np
import orjson
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ===================== CONFIG =====================

FORMAT_MANIFEST = Path("/home/arlo/gcs-bucket/Manifests/format_manifest.json")
GCS_ROOT = Path("/home/arlo/gcs-bucket")
CONDITIONING_ROOT = GCS_ROOT / "Conditioning"
PROGRESS_FILE = Path("/home/arlo/Data/amp_extraction_progress.txt")

SAMPLE_RATE = 44100
HOP_LENGTH = 4096
N_FFT = 8192
MIN_DURATION_SEC = 1.0  # Skip files shorter than this

# Batch save: accumulate results and write in batches to reduce GCS write overhead
SAVE_BATCH_SIZE = 500

# Lazy imports
_torch = None
_torchaudio = None


def get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def get_torchaudio():
    global _torchaudio
    if _torchaudio is None:
        import torchaudio
        _torchaudio = torchaudio
    return _torchaudio


# ===================== PATH MAPPING =====================

def audio_to_amp_path(audio_rel_path):
    """Convert relative audio path to .amp.npy conditioning path.

    Audio:  protools/2025-03-28/.../HH.04_22.wav
    Amp:    /home/arlo/gcs-bucket/Conditioning/protools/2025-03-28/.../HH.04_22.amp.npy
    """
    stem = os.path.splitext(audio_rel_path)[0]
    return CONDITIONING_ROOT / f"{stem}.amp.npy"


# ===================== EXTRACTION =====================

def extract_amp(audio_path):
    """Extract amplitude envelope from audio file.

    Returns numpy float32 array of shape (n_frames,), normalized to [0, 1].
    Uses RMS envelope with window=N_FFT, hop=HOP_LENGTH (~10.77 Hz frame rate).
    """
    torch = get_torch()
    torchaudio = get_torchaudio()

    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.float()

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav[0]

    # Resample to 44100 if needed
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    # Pad if too short for one frame
    if wav.numel() < N_FFT:
        wav = torch.nn.functional.pad(wav, (0, N_FFT - wav.numel()))

    # RMS envelope
    frames = wav.unfold(0, N_FFT, HOP_LENGTH)  # (n_frames, N_FFT)
    amp = torch.sqrt((frames ** 2).mean(dim=1) + 1e-12)

    # Normalize to [0, 1]
    if amp.max() > 0:
        amp = amp / amp.max()

    return amp.numpy().astype(np.float32)


def process_single(entry):
    """Process one entry: extract amp and return (amp_path, amp_data) or error."""
    rel_path = entry['path']
    audio_path = GCS_ROOT / rel_path
    amp_path = audio_to_amp_path(rel_path)

    # Skip if amp already exists
    if amp_path.exists():
        return {"status": "exists", "path": rel_path}

    if not audio_path.exists():
        return {"status": "missing", "path": rel_path}

    try:
        # Check duration first (cheap metadata read)
        torch = get_torch()
        torchaudio = get_torchaudio()
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        if duration < MIN_DURATION_SEC:
            return {"status": "too_short", "path": rel_path, "duration": duration}

        # Extract
        amp = extract_amp(audio_path)

        return {
            "status": "ok",
            "path": rel_path,
            "amp_path": str(amp_path),
            "amp_data": amp,
            "frames": len(amp),
        }

    except Exception as e:
        return {"status": "error", "path": rel_path, "error": str(e)}


def save_batch(batch, progress_lock, progress_file):
    """Save a batch of extracted amp arrays to disk and log to progress file."""
    saved = 0
    saved_paths = []
    for item in batch:
        try:
            amp_path = Path(item["amp_path"])
            amp_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(amp_path), item["amp_data"])
            saved += 1
            saved_paths.append(item["path"])
        except Exception as e:
            logging.warning(f"Failed to save {item['amp_path']}: {e}")
    # Append all successfully saved paths to progress file
    if saved_paths:
        with progress_lock:
            with open(progress_file, 'a') as f:
                f.write('\n'.join(saved_paths) + '\n')
    return saved


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="Batch extract amplitude envelopes")
    parser.add_argument("--workers", type=int, default=32, help="Thread count")
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Count only, don't extract")
    parser.add_argument("--batch-size", type=int, default=SAVE_BATCH_SIZE, help="Save batch size")
    parser.add_argument("--reset", action="store_true", help="Clear progress file and start fresh")
    args = parser.parse_args()

    if args.reset and PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logging.info(f"Cleared progress file: {PROGRESS_FILE}")

    # Load format manifest
    logging.info(f"Loading {FORMAT_MANIFEST.name}...")
    with open(FORMAT_MANIFEST, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = manifest['entries']
    logging.info(f"Total entries: {len(entries):,}")
    logging.info(f"Already have conditioning: {manifest['stats']['with_conditioning']:,}")
    logging.info(f"Need conditioning: {manifest['stats']['needs_conditioning']:,}")

    # Load progress file for resume support
    already_done = set()
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    already_done.add(line)
        logging.info(f"Resuming: {len(already_done):,} entries already processed (from {PROGRESS_FILE.name})")

    # Filter to entries that need amp extraction
    # Skip entries that already have conditioning, are too short, or already processed
    to_process = []
    skipped_manifest = 0
    skipped_progress = 0
    for e in entries:
        if e.get('has_conditioning'):
            skipped_manifest += 1
            continue
        if e.get('has_latent') == 'skipped':
            skipped_manifest += 1
            continue
        if e['path'] in already_done:
            skipped_progress += 1
            continue
        to_process.append(e)

    logging.info(f"Skipped (manifest): {skipped_manifest:,}")
    logging.info(f"Skipped (already processed): {skipped_progress:,}")
    logging.info(f"Entries to process: {len(to_process):,}")

    if args.limit > 0:
        to_process = to_process[:args.limit]
        logging.info(f"Limited to: {len(to_process):,}")

    if args.dry_run:
        logging.info("Dry run — exiting.")
        return

    # Process with thread pool + async save thread
    logging.info(f"Extracting with {args.workers} workers, batch save every {args.batch_size}...")

    NUM_SAVE_WORKERS = 8
    stats = {"ok": 0, "exists": 0, "missing": 0, "too_short": 0, "error": 0, "saved": 0}
    save_buffer = []
    progress_lock = Lock()
    save_queue = Queue()
    skip_buffer = []  # Non-ok paths to log so they aren't retried

    def save_worker():
        """Background thread that saves batches from the queue."""
        while True:
            batch = save_queue.get()
            if batch is None:  # Poison pill
                break
            saved = save_batch(batch, progress_lock, PROGRESS_FILE)
            with progress_lock:
                stats["saved"] += saved

    save_threads = []
    for _ in range(NUM_SAVE_WORKERS):
        t = Thread(target=save_worker, daemon=True)
        t.start()
        save_threads.append(t)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_single, e): e for e in to_process}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting amp", smoothing=0.1):
            result = future.result()
            status = result["status"]
            stats[status] = stats.get(status, 0) + 1

            if status == "ok":
                save_buffer.append(result)
                if len(save_buffer) >= args.batch_size:
                    save_queue.put(save_buffer)
                    save_buffer = []
            elif status in ("missing", "too_short", "exists"):
                skip_buffer.append(result["path"])
                if len(skip_buffer) >= args.batch_size:
                    with progress_lock:
                        with open(PROGRESS_FILE, 'a') as f:
                            f.write('\n'.join(skip_buffer) + '\n')
                    skip_buffer.clear()

    # Flush remaining
    if save_buffer:
        save_queue.put(save_buffer)

    # Signal all save threads to finish and wait
    logging.info(f"Extraction done. Waiting for {NUM_SAVE_WORKERS} save workers to flush (~{save_queue.qsize()} batches remaining)...")
    for _ in range(NUM_SAVE_WORKERS):
        save_queue.put(None)  # One poison pill per worker
    for t in save_threads:
        t.join()

    # Flush remaining skip buffer
    if skip_buffer:
        with open(PROGRESS_FILE, 'a') as f:
            f.write('\n'.join(skip_buffer) + '\n')

    # Summary
    logging.info("")
    logging.info("=" * 50)
    logging.info("EXTRACTION SUMMARY")
    logging.info("=" * 50)
    for k, v in sorted(stats.items()):
        logging.info(f"  {k:15s}: {v:>8,}")
    logging.info(f"  {'total':15s}: {len(to_process):>8,}")


if __name__ == "__main__":
    main()
