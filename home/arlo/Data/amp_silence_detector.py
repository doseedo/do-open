#!/usr/bin/env python3
"""
Amp-Based Silence Detector

Two-phase silence detection using pre-extracted .amp.npy amplitude envelopes:

Phase 1 — Amp scan (fast, all files):
  Loads normalized [0,1] RMS envelopes and classifies by amplitude statistics.
  Detects silent regions, flatlines, room tone, and flags constant-energy suspects.

Phase 2 — Spectral flatness (suspects only, ~10% of files):
  Loads raw audio for constant-energy suspects and computes spectral flatness.
  Separates noise/hiss (flat spectrum) from actual dense content (tonal).

Classifications:
  fully_silent        — Verified inaudible (abs_peak < 0.01 after Phase 2 check)
  mostly_silent       — >70% silence ratio with low energy
  noise_hiss          — Constant energy + flat spectrum (broadband noise)
  has_silent_regions  — Contains silent gaps >= 2s
  normal              — No significant silence

Usage:
    python3 amp_silence_detector.py --workers 64
    python3 amp_silence_detector.py --workers 64 --skip-spectral   # Phase 1 only
    python3 amp_silence_detector.py --mode validate                # GT comparison only
    python3 amp_silence_detector.py --mode stats                   # Print existing results
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# ===================== CONFIG =====================

GCS_ROOT = Path("/home/arlo/gcs-bucket")
CONDITIONING_ROOT = GCS_ROOT / "Conditioning"
FORMAT_MANIFEST = GCS_ROOT / "Manifests" / "format_manifest.json"
CONSOLIDATED_MANIFEST = GCS_ROOT / "Manifests" / "consolidated_manifest.json"

OUTPUT_DIR = Path("/home/arlo/Data/silence_detector")
OUTPUT_FILE = OUTPUT_DIR / "amp_silence_results.json"
SILENT_FILES_OUTPUT = OUTPUT_DIR / "silent_files.json"

AMP_FPS = 44100 / 4096  # ~10.77 Hz

# Phase 1 thresholds (amp envelope, normalized [0,1])
SILENT_FRAME_THRESH = 0.005       # Frame amplitude below this = silent
MIN_SILENCE_DURATION_SEC = 0.5    # Ignore silent regions shorter than this
GAP_MERGE_SEC = 0.3               # Merge regions closer than this
SIGNIFICANT_SILENCE_SEC = 2.0     # Threshold for "has_silent_regions"

# Flatline / room tone thresholds
FLATLINE_STD = 0.02               # std below this = flatline
FLATLINE_MEAN = 0.85              # mean above this = flatline (normalized noise)
ROOM_TONE_P95 = 0.05             # p95 below this = room tone (one spike, rest near zero)

# Mostly silent thresholds
MOSTLY_SILENT_RATIO = 0.70        # silence ratio above this
MOSTLY_SILENT_MEAN = 0.05         # amp mean below this

# Constant energy suspect thresholds (flagged for Phase 2)
SUSPECT_MIN = 0.3                 # amp.min() above this (never drops to silence)
SUSPECT_P5 = 0.4                  # p5 above this

# Phase 2 threshold (spectral flatness)
SPECTRAL_FLATNESS_THRESH = 0.3    # Above this = broadband noise


# ===================== PHASE 1: AMP ANALYSIS =====================

def find_silent_regions(amp, fps):
    """Find contiguous silent regions in amplitude envelope.

    Returns list of (start_sec, end_sec, duration_sec) tuples.
    """
    silent_mask = amp < SILENT_FRAME_THRESH
    T = len(silent_mask)
    if T == 0:
        return [], 0.0

    min_frames = max(1, int(MIN_SILENCE_DURATION_SEC * fps))
    gap_frames = max(1, int(GAP_MERGE_SEC * fps))

    # Find contiguous runs
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
        return [], 0.0

    # Merge close gaps
    merged = [raw_regions[0]]
    for s, e in raw_regions[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_frames:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    # Filter by minimum duration
    filtered = [(s, e) for s, e in merged if (e - s) >= min_frames]

    # Convert to seconds
    regions = []
    total_silent = 0.0
    for s, e in filtered:
        start_sec = round(s / fps, 2)
        end_sec = round(e / fps, 2)
        dur = round(end_sec - start_sec, 2)
        total_silent += dur
        regions.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": dur,
        })

    silence_ratio = total_silent / (T / fps) if T > 0 else 0.0
    return regions, round(silence_ratio, 4)


def classify_amp(amp, regions, silence_ratio):
    """Classify file based on amp envelope statistics.

    Returns (classification, stats_dict).
    classification is one of: fully_silent_suspect, mostly_silent,
    constant_energy_suspect, has_silent_regions, normal.

    Note: fully_silent_suspect is verified in Phase 2 via absolute levels.
    Per-file normalization means we can't distinguish truly silent files
    from sparse-but-real audio using only the amp envelope.
    """
    std = float(np.std(amp))
    mean = float(np.mean(amp))
    amp_min = float(amp.min())
    amp_max = float(amp.max())
    p5 = float(np.percentile(amp, 5))
    p95 = float(np.percentile(amp, 95))

    stats = {
        "std": round(std, 6),
        "mean": round(mean, 6),
        "min": round(amp_min, 6),
        "max": round(amp_max, 6),
        "p5": round(p5, 6),
        "p95": round(p95, 6),
    }

    # Room tone candidate: nearly all frames near zero, one spike drives max to 1.0
    # BUT per-file normalization means sparse real audio also looks like this.
    # Route to Phase 2 for absolute level verification.
    if p95 < ROOM_TONE_P95:
        return "fully_silent_suspect", stats

    # Flatline OR constant energy: route to Phase 2 spectral verification
    # Flatline: very low variation + high mean = normalized noise or steady content
    # Constant energy: never drops to silence at all
    if (std < FLATLINE_STD and mean > FLATLINE_MEAN) or \
       (amp_min > SUSPECT_MIN and p5 > SUSPECT_P5):
        return "constant_energy_suspect", stats

    # Mostly silent: high silence ratio with very low mean
    if silence_ratio > MOSTLY_SILENT_RATIO and mean < MOSTLY_SILENT_MEAN:
        return "mostly_silent", stats

    # Has significant silent regions
    for r in regions:
        if r["duration_sec"] >= SIGNIFICANT_SILENCE_SEC:
            return "has_silent_regions", stats

    return "normal", stats


def process_amp_file(entry):
    """Phase 1 worker: analyze one amp file."""
    rel_path = entry["path"]
    stem = os.path.splitext(rel_path)[0]
    amp_path = str(CONDITIONING_ROOT / f"{stem}.amp.npy")
    audio_path = str(GCS_ROOT / rel_path)

    try:
        amp = np.load(amp_path)
    except FileNotFoundError:
        return {"status": "no_amp", "path": rel_path}
    except Exception as e:
        return {"status": "error", "path": rel_path, "error": str(e)}

    if len(amp) < 5:
        return {"status": "too_short", "path": rel_path}

    duration = len(amp) / AMP_FPS
    regions, silence_ratio = find_silent_regions(amp, AMP_FPS)
    classification, stats = classify_amp(amp, regions, silence_ratio)

    result = {
        "status": "ok",
        "path": rel_path,
        "audio_path": audio_path,
        "classification": classification,
        "silence_ratio": silence_ratio,
        "duration_sec": round(duration, 2),
        "n_frames": len(amp),
        **stats,
    }

    if classification != "normal":
        result["silent_regions"] = regions
        result["num_silent_regions"] = len(regions)

    return result


# ===================== PHASE 2: SPECTRAL FLATNESS =====================

def compute_spectral_flatness(audio_path):
    """Compute spectral flatness, centroid, and absolute levels for an audio file.

    Returns (flatness, centroid, abs_peak, abs_rms, error).
    """
    import torchaudio

    try:
        wav, sr = torchaudio.load(audio_path)
    except Exception as e:
        return None, None, None, None, str(e)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0)
    else:
        wav = wav[0]

    audio = wav.numpy().astype(np.float64)
    N = len(audio)

    abs_peak = float(np.max(np.abs(audio)))
    abs_rms = float(np.sqrt(np.mean(audio ** 2)))

    n_fft = min(8192, N)
    if N < n_fft:
        return None, None, abs_peak, abs_rms, "too_short"

    hop = n_fft // 2
    n_windows = max(1, (N - n_fft) // hop)
    window = np.hanning(n_fft)

    psd_sum = np.zeros(n_fft // 2 + 1)
    for i in range(n_windows):
        start = i * hop
        chunk = audio[start:start + n_fft]
        spectrum = np.abs(np.fft.rfft(chunk * window)) ** 2
        psd_sum += spectrum
    psd = psd_sum / max(n_windows, 1)

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Spectral flatness: geometric mean / arithmetic mean
    psd_pos = psd[psd > 0]
    if len(psd_pos) < 10:
        return 0.0, 0.0, abs_peak, abs_rms, None

    geo_mean = np.exp(np.mean(np.log(psd_pos + 1e-20)))
    arith_mean = np.mean(psd_pos)
    flatness = float(geo_mean / arith_mean) if arith_mean > 0 else 0.0

    # Spectral centroid
    total_energy = psd.sum()
    centroid = float(np.sum(freqs * psd) / total_energy) if total_energy > 0 else 0.0

    return flatness, centroid, abs_peak, abs_rms, None


def process_suspect(result):
    """Phase 2 worker: run absolute level + spectral analysis on suspects.

    Handles both fully_silent_suspect and constant_energy_suspect.

    For fully_silent_suspect (from p95 < 0.05):
      - abs_peak < 0.01 → fully_silent (truly inaudible)
      - abs_peak >= 0.01 → has_silent_regions (sparse but real audio)

    For constant_energy_suspect (flatline / never-drops-to-zero):
      - spectral flatness > 0.3 → noise_hiss (broadband noise)
      - abs_peak < 0.01 → fully_silent (inaudible)
      - Otherwise → normal (real dense content)
    """
    audio_path = result["audio_path"]
    original_cls = result["classification"]
    flatness, centroid, abs_peak, abs_rms, error = compute_spectral_flatness(audio_path)

    if abs_peak is not None:
        result["abs_peak"] = round(abs_peak, 6)
    if abs_rms is not None:
        result["abs_rms"] = round(abs_rms, 6)

    # --- fully_silent_suspect: just need absolute levels ---
    if original_cls == "fully_silent_suspect":
        if abs_peak is not None and abs_peak < 0.01:
            result["classification"] = "fully_silent"
        else:
            # Real audio exists — reclassify as has_silent_regions
            result["classification"] = "has_silent_regions"
        return result

    # --- constant_energy_suspect: full spectral analysis ---
    if error:
        result["spectral_error"] = error
        if abs_peak is not None and abs_peak < 0.01:
            result["classification"] = "fully_silent"
        else:
            result["classification"] = "normal"
        return result

    result["spectral_flatness"] = round(flatness, 4)
    result["spectral_centroid"] = round(centroid, 0)

    if flatness > SPECTRAL_FLATNESS_THRESH:
        result["classification"] = "noise_hiss"
    elif abs_peak < 0.01:
        result["classification"] = "fully_silent"
    else:
        result["classification"] = "normal"

    return result


# ===================== GT VALIDATION =====================

def validate_against_gt(results_by_path, gt_silent_paths):
    """Compare detections against GT silent labels from consolidated manifest."""
    # "detected as silent" = fully_silent, mostly_silent, or noise_hiss
    SILENT_CLASSES = {"fully_silent", "mostly_silent", "noise_hiss"}

    detected_silent = set()
    for path, r in results_by_path.items():
        if r.get("classification") in SILENT_CLASSES:
            detected_silent.add(r.get("audio_path", ""))

    gt_count = len(gt_silent_paths)
    detected_count = len(detected_silent)

    tp = len(gt_silent_paths & detected_silent)
    fn = gt_count - tp
    fp = detected_count - tp

    precision = tp / detected_count if detected_count > 0 else 0.0
    recall = tp / gt_count if gt_count > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Detail on each GT file
    gt_details = []
    for gt_path in sorted(gt_silent_paths):
        # Find by audio_path match
        found = None
        for path, r in results_by_path.items():
            if r.get("audio_path") == gt_path:
                found = r
                break

        detail = {"audio_path": gt_path, "filename": os.path.basename(gt_path)}
        if found:
            detail["detected_as"] = found.get("classification", "?")
            detail["silence_ratio"] = found.get("silence_ratio", 0)
            detail["std"] = found.get("std", 0)
            detail["mean"] = found.get("mean", 0)
            detail["p95"] = found.get("p95", 0)
            detail["spectral_flatness"] = found.get("spectral_flatness")
            detail["hit"] = found.get("classification") in SILENT_CLASSES
        else:
            detail["detected_as"] = "not_processed"
            detail["hit"] = False

        gt_details.append(detail)

    logging.info("=" * 60)
    logging.info("GT VALIDATION")
    logging.info("=" * 60)
    logging.info(f"GT silent files:      {gt_count}")
    logging.info(f"Detected silent:      {detected_count}")
    logging.info(f"True positives:       {tp}")
    logging.info(f"False negatives:      {fn}")
    logging.info(f"False positives:      {fp}")
    logging.info(f"Precision:            {precision:.4f}")
    logging.info(f"Recall:               {recall:.4f}")
    logging.info(f"F1:                   {f1:.4f}")
    logging.info("")
    for d in gt_details:
        status = "HIT" if d["hit"] else "MISS"
        logging.info(f"  [{status}] {d['filename']} → {d['detected_as']}"
                     f" (ratio={d.get('silence_ratio', '?')}, p95={d.get('p95', '?')},"
                     f" flatness={d.get('spectral_flatness', 'n/a')})")

    return {
        "gt_silent_count": gt_count,
        "detected_silent_count": detected_count,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gt_details": gt_details,
    }


# ===================== OUTPUT =====================

def save_silent_files_json(results):
    """Write silent_files.json compatible with monitor_service get_silent_files()."""
    silent = []
    near_silent = []

    for r in results:
        cls = r.get("classification")
        entry = {
            "path": r["audio_path"],
            "classification": cls,
            "silence_ratio": r.get("silence_ratio", 0),
        }

        if cls in ("fully_silent", "noise_hiss"):
            silent.append(entry)
        elif cls == "mostly_silent":
            near_silent.append(entry)

    data = {
        "silent": silent,
        "near_silent": near_silent,
        "source": "amp_silence_detector",
        "generated_at": datetime.now().isoformat(),
        "counts": {
            "silent": len(silent),
            "near_silent": len(near_silent),
        },
    }

    with open(SILENT_FILES_OUTPUT, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    logging.info(f"Wrote {SILENT_FILES_OUTPUT}: {len(silent)} silent, {len(near_silent)} near_silent")


# ===================== STATS MODE =====================

def print_stats():
    """Print summary from existing results file."""
    if not OUTPUT_FILE.exists():
        logging.error(f"No results at {OUTPUT_FILE}")
        return

    with open(OUTPUT_FILE, 'rb') as f:
        data = orjson.loads(f.read())

    summary = data.get("summary", {})
    validation = data.get("validation", {})
    total = data.get("total_processed", 0)

    logging.info("=" * 60)
    logging.info("AMP SILENCE DETECTION SUMMARY")
    logging.info("=" * 60)
    for key, count in sorted(summary.items()):
        pct = (count / total * 100) if total > 0 else 0
        logging.info(f"  {key:25s}: {count:>8,d} ({pct:5.2f}%)")
    logging.info(f"  {'TOTAL':25s}: {total:>8,d}")

    if validation:
        logging.info("")
        logging.info("GT VALIDATION")
        logging.info(f"  Precision: {validation.get('precision', 0):.4f}")
        logging.info(f"  Recall:    {validation.get('recall', 0):.4f}")
        logging.info(f"  F1:        {validation.get('f1', 0):.4f}")

    results = data.get("results", [])
    if results:
        # Classification breakdown for non-normal
        from collections import Counter
        cls_counts = Counter(r["classification"] for r in results)
        logging.info("")
        logging.info("NON-NORMAL BREAKDOWN:")
        for cls, cnt in cls_counts.most_common():
            logging.info(f"  {cls:25s}: {cnt:>8,d}")


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser(description="Amp-based silence detection")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0, help="Limit entries (0=all)")
    parser.add_argument("--skip-spectral", action="store_true", help="Skip Phase 2 spectral analysis")
    parser.add_argument("--mode", choices=["detect", "validate", "stats"], default="detect")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "stats":
        print_stats()
        return

    # Load GT labels
    logging.info(f"Loading consolidated manifest for GT labels...")
    with open(CONSOLIDATED_MANIFEST, 'rb') as f:
        consolidated = orjson.loads(f.read())
    gt_silent_paths = {
        e["audio_path"] for e in consolidated["entries"]
        if e.get("group") == "silent"
    }
    logging.info(f"GT silent files: {len(gt_silent_paths)}")

    if args.mode == "validate":
        if not OUTPUT_FILE.exists():
            logging.error(f"No results at {OUTPUT_FILE}. Run --mode detect first.")
            return
        with open(OUTPUT_FILE, 'rb') as f:
            data = orjson.loads(f.read())
        results_by_path = {r["path"]: r for r in data.get("results", [])}
        validation = validate_against_gt(results_by_path, gt_silent_paths)
        data["validation"] = validation
        with open(OUTPUT_FILE, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))
        return

    # === PHASE 1: Amp scan ===
    logging.info(f"Loading {FORMAT_MANIFEST.name}...")
    with open(FORMAT_MANIFEST, 'rb') as f:
        manifest = orjson.loads(f.read())

    entries = manifest["entries"]
    logging.info(f"Total entries: {len(entries):,}")

    # Filter to entries that should have amp files
    to_process = [e for e in entries if e.get("has_latent") != "skipped"]
    logging.info(f"Entries to scan: {len(to_process):,}")

    if args.limit > 0:
        to_process = to_process[:args.limit]
        logging.info(f"Limited to: {len(to_process):,}")

    logging.info(f"Phase 1: Scanning amp envelopes with {args.workers} workers...")

    summary = {
        "fully_silent": 0,
        "fully_silent_suspect": 0,
        "mostly_silent": 0,
        "noise_hiss": 0,
        "has_silent_regions": 0,
        "constant_energy_suspect": 0,
        "normal": 0,
        "no_amp": 0,
        "too_short": 0,
        "error": 0,
    }
    all_results = []
    suspects = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_amp_file, e): e for e in to_process}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Phase 1: Amp scan", smoothing=0.05):
            result = future.result()
            status = result.get("status")

            if status != "ok":
                summary[status] = summary.get(status, 0) + 1
                continue

            cls = result["classification"]
            summary[cls] = summary.get(cls, 0) + 1

            if cls in ("constant_energy_suspect", "fully_silent_suspect"):
                suspects.append(result)
            elif cls != "normal":
                all_results.append(result)

    logging.info("")
    logging.info("Phase 1 complete:")
    for key, count in sorted(summary.items()):
        logging.info(f"  {key:30s}: {count:>8,d}")

    # === PHASE 2: Absolute level + spectral analysis on suspects ===
    if suspects and not args.skip_spectral:
        n_silent_suspects = sum(1 for s in suspects if s["classification"] == "fully_silent_suspect")
        n_energy_suspects = sum(1 for s in suspects if s["classification"] == "constant_energy_suspect")
        logging.info(f"\nPhase 2: Verifying {len(suspects):,} suspects "
                     f"({n_silent_suspects:,} fully_silent_suspect, "
                     f"{n_energy_suspects:,} constant_energy_suspect)...")

        # Use fewer workers for audio loading (heavier I/O)
        spectral_workers = min(args.workers, 16)
        reclassified = {}

        # Remember original classification before Phase 2 overwrites it
        for s in suspects:
            s["_phase1_cls"] = s["classification"]

        with ThreadPoolExecutor(max_workers=spectral_workers) as executor:
            futures = {executor.submit(process_suspect, s): s for s in suspects}
            for future in tqdm(as_completed(futures), total=len(futures),
                               desc="Phase 2: Verify", smoothing=0.1):
                result = future.result()
                new_cls = result["classification"]
                reclassified[new_cls] = reclassified.get(new_cls, 0) + 1

                if new_cls != "normal":
                    all_results.append(result)

        # Rebuild summary: subtract Phase 1 suspect counts, add Phase 2 final counts
        summary["fully_silent_suspect"] = 0
        summary["constant_energy_suspect"] = 0
        for cls, count in reclassified.items():
            summary[cls] = summary.get(cls, 0) + count

        logging.info(f"Phase 2 reclassified: {reclassified}")
    elif suspects and args.skip_spectral:
        logging.info(f"Skipping Phase 2 (--skip-spectral). {len(suspects)} suspects left as-is.")
        # Keep suspects in results for later spectral pass
        all_results.extend(suspects)

    # Sort by silence ratio (most silent first)
    all_results.sort(key=lambda x: -(x.get("silence_ratio", 0)))

    total_processed = sum(v for k, v in summary.items()
                         if k not in ("no_amp", "error", "too_short"))

    # === PHASE 3: GT Validation ===
    results_by_path = {r["path"]: r for r in all_results}
    validation = validate_against_gt(results_by_path, gt_silent_paths)

    # === PHASE 4: Save outputs ===
    output_data = {
        "total_processed": total_processed,
        "summary": summary,
        "validation": validation,
        "parameters": {
            "amp_fps": AMP_FPS,
            "silent_frame_thresh": SILENT_FRAME_THRESH,
            "min_silence_duration_sec": MIN_SILENCE_DURATION_SEC,
            "gap_merge_sec": GAP_MERGE_SEC,
            "significant_silence_sec": SIGNIFICANT_SILENCE_SEC,
            "flatline_std": FLATLINE_STD,
            "flatline_mean": FLATLINE_MEAN,
            "room_tone_p95": ROOM_TONE_P95,
            "mostly_silent_ratio": MOSTLY_SILENT_RATIO,
            "mostly_silent_mean": MOSTLY_SILENT_MEAN,
            "suspect_min": SUSPECT_MIN,
            "suspect_p5": SUSPECT_P5,
            "spectral_flatness_thresh": SPECTRAL_FLATNESS_THRESH,
        },
        "results": all_results,
        "detected_at": datetime.now().isoformat(),
    }

    with open(OUTPUT_FILE, 'wb') as f:
        f.write(orjson.dumps(output_data, option=orjson.OPT_INDENT_2))
    logging.info(f"\nResults saved to {OUTPUT_FILE}")
    logging.info(f"Non-normal entries: {len(all_results):,}")

    # Write silent_files.json for monitor_service
    save_silent_files_json(all_results)

    # Final summary
    logging.info("")
    logging.info("=" * 60)
    logging.info("FINAL SUMMARY")
    logging.info("=" * 60)
    for key, count in sorted(summary.items()):
        if count > 0:
            pct = (count / total_processed * 100) if total_processed > 0 else 0
            logging.info(f"  {key:30s}: {count:>8,d} ({pct:5.2f}%)")
    logging.info(f"  {'TOTAL PROCESSED':30s}: {total_processed:>8,d}")


if __name__ == "__main__":
    main()
