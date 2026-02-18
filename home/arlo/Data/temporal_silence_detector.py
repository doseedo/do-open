#!/usr/bin/env python3
"""
Temporal Silence Detector (Audio-based)

Detects silent regions within audio files by computing frame-level RMS energy
directly from audio waveforms. The DCAE latent space does NOT encode silence
as low energy, so latent-based detection is unreliable.

Detects:
  - Fully silent files (>95% silence)
  - Mostly silent files (>70% silence)
  - Files with significant silent regions (>2s of silence anywhere)
  - Normal files (no significant silence)

Produces timestamp annotations for each silent region.

Usage:
    python3 temporal_silence_detector.py --mode detect --workers 32
    python3 temporal_silence_detector.py --mode validate
    python3 temporal_silence_detector.py --mode stats
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

# ===================== CONSTANTS =====================

MANIFEST_FILE = Path("/home/arlo/gcs-bucket/Manifests/consolidated_manifest.json")
OUTPUT_DIR = Path("/home/arlo/Data/silence_detector")
OUTPUT_FILE = OUTPUT_DIR / "temporal_silence_results.json"

# Audio analysis parameters — NO downsampling, work at native sample rate
TARGET_FPS = 43                  # Target frames per second (~23ms per frame)
# Frame size computed per-file: sr // TARGET_FPS

# Silence thresholds (audio RMS per frame)
SILENCE_RMS_THRESHOLD = 0.003    # Frame RMS below this = candidate silent
SILENCE_PEAK_THRESHOLD = 0.02    # Frame peak below this AND RMS below threshold = silent
                                 # Prevents transient-heavy files (kicks) from being missed

# Region parameters
MIN_SILENCE_DURATION_SEC = 0.5   # Ignore silent regions shorter than this
GAP_MERGE_SEC = 0.3              # Merge regions separated by less than this
SIGNIFICANT_SILENCE_SEC = 2.0    # Threshold for "has_silent_regions" classification

# Lazy imports
_torchaudio = None
_torch = None


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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


# ===================== CORE ANALYSIS =====================

def compute_frame_stats(audio_mono, frame_size):
    """Compute per-frame RMS and peak from mono audio [N] -> ([T], [T]).

    Returns (rms_array, peak_array) as numpy arrays.
    Works at full sample rate — no downsampling.
    """
    N = audio_mono.shape[0]
    n_frames = N // frame_size
    if n_frames == 0:
        return np.array([0.0]), np.array([0.0])

    # Reshape into frames
    frames = audio_mono[:n_frames * frame_size].reshape(n_frames, frame_size)
    # Use float64 to avoid overflow with high-amplitude int-like samples
    frames_f64 = frames.astype(np.float64)
    rms = np.sqrt(np.mean(frames_f64 ** 2, axis=1)).astype(np.float32)
    peak = np.max(np.abs(frames), axis=1)
    return rms, peak


def detect_silent_frames(rms, peak):
    """Dual-threshold silence detection using RMS AND peak.

    A frame is silent only if BOTH:
      - RMS < SILENCE_RMS_THRESHOLD (low average energy)
      - Peak < SILENCE_PEAK_THRESHOLD (no transients in frame)

    This prevents percussive files (kicks, snares) from being classified
    as silent — they have low RMS between hits but high peaks during hits.
    """
    return (rms < SILENCE_RMS_THRESHOLD) & (peak < SILENCE_PEAK_THRESHOLD)


def frames_to_regions(silent_mask, min_frames, gap_frames):
    """Convert boolean frame mask to list of (start_frame, end_frame) regions.

    Applies minimum duration filter and gap merging.
    """
    T = len(silent_mask)
    if T == 0:
        return []

    # Find contiguous silent runs
    raw_regions = []
    in_region = False
    start = 0

    for i in range(T):
        if silent_mask[i] and not in_region:
            start = i
            in_region = True
        elif not silent_mask[i] and in_region:
            raw_regions.append((start, i))
            in_region = False
    if in_region:
        raw_regions.append((start, T))

    if not raw_regions:
        return []

    # Merge regions separated by small gaps
    merged = [raw_regions[0]]
    for start, end in raw_regions[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= gap_frames:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # Filter by minimum duration
    filtered = [(s, e) for s, e in merged if (e - s) >= min_frames]

    return filtered


def classify_file(silence_ratio, regions, rms_mean, peak):
    """Classify file based on silence analysis.

    Uses silence_ratio from frame analysis but also checks file-level
    RMS and peak to prevent transient-heavy files (kicks, snares) from
    being misclassified as silent.
    """
    # Guard: if the file has significant energy or peaks, it's not fully silent
    # A kick drum has peak 0.9 but 99% silent frames — it's not "silent"
    if silence_ratio > 0.95:
        if peak > 0.05 or rms_mean > 0.01:
            # Has real content despite sparse transients
            return "has_silent_regions"
        return "fully_silent"
    elif silence_ratio > 0.70:
        if peak > 0.1 or rms_mean > 0.02:
            return "has_silent_regions"
        return "mostly_silent"
    else:
        for r in regions:
            if r["duration_sec"] >= SIGNIFICANT_SILENCE_SEC:
                return "has_silent_regions"
        return "normal"


def process_single_file(audio_path):
    """Worker: analyze one audio file for silence.

    Works at full sample rate — no downsampling.
    Uses dual threshold (RMS + peak) to avoid false positives on transients.
    """
    torch = get_torch()
    torchaudio = get_torchaudio()

    if not os.path.exists(audio_path):
        return {"status": "missing", "audio_path": audio_path}

    try:
        wav, sr = torchaudio.load(audio_path)

        # Convert to mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0)
        else:
            wav = wav[0]

        audio_np = wav.numpy()
        N = len(audio_np)
        duration_sec = N / sr

        # Frame size based on native sample rate (~43 fps)
        frame_size = max(256, sr // TARGET_FPS)

        if N < frame_size * 3:
            return {"status": "too_short", "audio_path": audio_path}

        # Compute per-frame RMS and peak at full sample rate
        rms, frame_peaks = compute_frame_stats(audio_np, frame_size)
        n_frames = len(rms)
        fps = n_frames / duration_sec if duration_sec > 0 else TARGET_FPS

        # Detect silent frames using dual threshold
        silent_mask = detect_silent_frames(rms, frame_peaks)

        # Convert to regions
        min_frames = max(1, int(MIN_SILENCE_DURATION_SEC * fps))
        gap_frames = max(1, int(GAP_MERGE_SEC * fps))
        frame_regions = frames_to_regions(silent_mask, min_frames, gap_frames)

        # Convert frame regions to seconds
        regions = []
        total_silent_sec = 0.0
        for start_f, end_f in frame_regions:
            start_sec = round(start_f / fps, 2)
            end_sec = round(end_f / fps, 2)
            dur = round(end_sec - start_sec, 2)
            total_silent_sec += dur
            regions.append({
                "start_sec": start_sec,
                "end_sec": end_sec,
                "duration_sec": dur,
            })

        silence_ratio = total_silent_sec / duration_sec if duration_sec > 0 else 0.0
        # Stats
        rms_mean = float(np.mean(rms))
        rms_std = float(np.std(rms))
        rms_max = float(np.max(rms))
        file_peak = float(np.max(np.abs(audio_np)))

        classification = classify_file(silence_ratio, regions, rms_mean, file_peak)

        return {
            "status": "ok",
            "audio_path": audio_path,
            "classification": classification,
            "silence_ratio": round(silence_ratio, 4),
            "duration_sec": round(duration_sec, 2),
            "rms_mean": round(rms_mean, 6),
            "rms_std": round(rms_std, 6),
            "rms_max": round(rms_max, 6),
            "peak": round(file_peak, 6),
            "total_frames": n_frames,
            "silent_frames": int(np.sum(silent_mask)),
            "silent_regions": regions,
            "num_silent_regions": len(regions),
        }

    except Exception as e:
        return {"status": "error", "audio_path": audio_path, "error": str(e)}


# ===================== MANIFEST & I/O =====================

def load_manifest():
    """Load consolidated manifest."""
    logging.info(f"Loading manifest from {MANIFEST_FILE}...")
    with open(MANIFEST_FILE, 'rb') as f:
        data = orjson.loads(f.read())
    entries = data.get("entries", [])
    logging.info(f"Loaded {len(entries)} entries")
    return entries


def get_audio_paths(entries):
    """Extract audio paths from manifest entries."""
    paths = []
    for entry in entries:
        audio_path = entry.get("audio_path", "")
        if audio_path:
            paths.append(audio_path)
    return paths


def get_gt_silent_paths(entries):
    """Get set of audio paths labeled as 'silent' in the manifest."""
    return {
        e.get("audio_path", "")
        for e in entries
        if e.get("group") == "silent"
    }


# ===================== DETECTION =====================

def run_detection(paths, num_workers=32):
    """Run temporal silence detection on all audio files."""
    logging.info(f"Analyzing {len(paths)} files with {num_workers} threads...")

    results = []
    summary = {
        "fully_silent": 0,
        "mostly_silent": 0,
        "has_silent_regions": 0,
        "normal": 0,
        "errors": 0,
        "missing": 0,
        "too_short": 0,
    }

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for result in tqdm(
            executor.map(process_single_file, paths),
            total=len(paths),
            desc="Detecting silence",
            smoothing=0.1,
        ):
            status = result.get("status")
            if status == "ok":
                classification = result["classification"]
                summary[classification] = summary.get(classification, 0) + 1
                # Only store non-normal results in detail to save space
                if classification != "normal":
                    results.append(result)
            elif status == "missing":
                summary["missing"] += 1
            elif status == "too_short":
                summary["too_short"] += 1
            else:
                summary["errors"] += 1

    return results, summary


# ===================== VALIDATION =====================

def validate_against_gt(results, gt_silent_paths, summary):
    """Compare detection results against GT silent labels."""
    detected_silent_paths = set()
    detected_classifications = {}
    for r in results:
        path = r.get("audio_path", "")
        cls = r.get("classification", "normal")
        detected_classifications[path] = cls
        if cls in ("fully_silent", "mostly_silent"):
            detected_silent_paths.add(path)

    gt_count = len(gt_silent_paths)
    detected_count = len(detected_silent_paths)

    tp = len(gt_silent_paths & detected_silent_paths)
    fn = gt_count - tp
    fp = detected_count - tp

    precision = tp / detected_count if detected_count > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    false_negatives = []
    for path in gt_silent_paths:
        if path not in detected_silent_paths:
            cls = detected_classifications.get(path, "not_processed")
            detail = next((r for r in results if r.get("audio_path") == path), None)
            entry = {"audio_path": path, "detected_as": cls}
            if detail:
                entry["silence_ratio"] = detail.get("silence_ratio", 0)
                entry["rms_mean"] = detail.get("rms_mean", 0)
                entry["num_silent_regions"] = detail.get("num_silent_regions", 0)
            false_negatives.append(entry)

    # Print summary
    logging.info("=" * 60)
    logging.info("VALIDATION AGAINST GT LABELS")
    logging.info("=" * 60)
    logging.info(f"GT silent files:       {gt_count}")
    logging.info(f"Detected silent:       {detected_count}")
    logging.info(f"True positives:        {tp}")
    logging.info(f"False negatives:       {fn} (GT silent, missed)")
    logging.info(f"False positives:       {fp} (detected, not in GT)")
    logging.info(f"Precision:             {precision:.4f}")
    logging.info(f"Recall:                {recall:.4f}")
    logging.info(f"F1:                    {f1:.4f}")

    if false_negatives:
        logging.info("")
        logging.info("FALSE NEGATIVES (GT silent but not detected):")
        for fn_entry in false_negatives[:20]:
            logging.info(f"  {os.path.basename(fn_entry['audio_path'])} -> {fn_entry.get('detected_as', '?')}"
                         f" (silence_ratio={fn_entry.get('silence_ratio', '?')},"
                         f" rms={fn_entry.get('rms_mean', '?')})")

    validation = {
        "gt_silent_count": gt_count,
        "detected_silent_count": detected_count,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_negative_details": false_negatives,
    }

    return validation


# ===================== STATS =====================

def print_stats(output_path):
    """Print statistics from existing results file."""
    if not output_path.exists():
        logging.error(f"Results file not found: {output_path}")
        return

    with open(output_path, 'rb') as f:
        data = orjson.loads(f.read())

    summary = data.get("summary", {})
    validation = data.get("validation", {})
    results = data.get("results", [])

    logging.info("=" * 60)
    logging.info("DETECTION SUMMARY")
    logging.info("=" * 60)
    total = data.get("total_processed", 0)
    for key, count in summary.items():
        pct = (count / total * 100) if total > 0 else 0
        logging.info(f"  {key:25s}: {count:>8,d} ({pct:5.2f}%)")
    logging.info(f"  {'TOTAL':25s}: {total:>8,d}")

    if validation:
        logging.info("")
        logging.info("VALIDATION")
        logging.info(f"  GT silent:    {validation.get('gt_silent_count', 0)}")
        logging.info(f"  Precision:    {validation.get('precision', 0):.4f}")
        logging.info(f"  Recall:       {validation.get('recall', 0):.4f}")
        logging.info(f"  F1:           {validation.get('f1', 0):.4f}")

    if results:
        ratios = [r.get("silence_ratio", 0) for r in results]
        if ratios:
            logging.info("")
            logging.info("SILENCE RATIO DISTRIBUTION (non-normal files):")
            arr = np.array(ratios)
            for pct in [10, 25, 50, 75, 90]:
                logging.info(f"  P{pct:02d}: {np.percentile(arr, pct):.4f}")
            logging.info(f"  Mean: {arr.mean():.4f}, Std: {arr.std():.4f}")

        logging.info("")
        logging.info("TOP 20 FILES BY SILENCE RATIO:")
        sorted_results = sorted(results, key=lambda x: -x.get("silence_ratio", 0))
        for r in sorted_results[:20]:
            fname = os.path.basename(r.get("audio_path", ""))
            logging.info(f"  {r.get('silence_ratio', 0):.3f} | {r.get('classification', ''):20s} | "
                         f"{r.get('num_silent_regions', 0):2d} regions | rms={r.get('rms_mean', 0):.5f} | {fname}")


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="Temporal silence detection from audio waveforms")
    parser.add_argument("--mode", choices=["detect", "validate", "stats"],
                        default="detect", help="Mode of operation")
    parser.add_argument("--workers", type=int, default=32,
                        help="Number of threads (default: 32)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Limit number of files to process (0=all)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_FILE),
                        help=f"Output file path (default: {OUTPUT_FILE})")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.mode == "stats":
        print_stats(output_path)
        return

    # Load manifest
    entries = load_manifest()
    gt_silent_paths = get_gt_silent_paths(entries)
    logging.info(f"GT silent files: {len(gt_silent_paths)}")

    if args.mode == "validate":
        if not output_path.exists():
            logging.error(f"No results file found at {output_path}. Run --mode detect first.")
            return
        with open(output_path, 'rb') as f:
            data = orjson.loads(f.read())
        results = data.get("results", [])
        summary = data.get("summary", {})
        validation = validate_against_gt(results, gt_silent_paths, summary)
        data["validation"] = validation
        with open(output_path, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        logging.info(f"Updated validation in {output_path}")
        return

    # Mode: detect
    paths = get_audio_paths(entries)
    logging.info(f"Total audio files: {len(paths)}")

    if args.limit > 0:
        paths = paths[:args.limit]
        logging.info(f"Limited to {len(paths)} files")

    results, summary = run_detection(paths, num_workers=args.workers)

    # Print summary
    logging.info("")
    logging.info("DETECTION SUMMARY:")
    total = len(paths)
    for key, count in summary.items():
        pct = (count / total * 100) if total > 0 else 0
        logging.info(f"  {key:25s}: {count:>8,d} ({pct:5.2f}%)")

    # Validate against GT
    validation = validate_against_gt(results, gt_silent_paths, summary)

    # Save results
    output_data = {
        "total_processed": total,
        "summary": summary,
        "validation": validation,
        "parameters": {
            "target_fps": TARGET_FPS,
            "silence_rms_threshold": SILENCE_RMS_THRESHOLD,
            "silence_peak_threshold": SILENCE_PEAK_THRESHOLD,
            "min_silence_duration_sec": MIN_SILENCE_DURATION_SEC,
            "gap_merge_sec": GAP_MERGE_SEC,
            "significant_silence_sec": SIGNIFICANT_SILENCE_SEC,
        },
        "results": sorted(results, key=lambda x: -x.get("silence_ratio", 0)),
        "detected_at": datetime.now().isoformat(),
    }

    with open(output_path, 'wb') as f:
        f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))

    logging.info(f"\nResults saved to {output_path}")
    logging.info(f"Total non-normal files: {len(results)}")


if __name__ == "__main__":
    main()
