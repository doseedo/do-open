#!/usr/bin/env python3
"""
Recompute silent regions for files where over-merging caused inaccurate overlays.

Only reprocesses ~2.7K files where silence_ratio > 0.95 but abs_peak > 0.01
(real audio exists but regions were merged across transients).
Reads .amp.npy via gcsfuse — 2.7K files is fine even over FUSE.
"""

import os
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import orjson
from tqdm import tqdm

# ===================== CONFIG =====================

GCS_ROOT = Path("/home/arlo/gcs-bucket")
CONDITIONING_ROOT = GCS_ROOT / "Conditioning"

INPUT_FILE = Path("/home/arlo/Data/silence_detector/amp_silence_results.json")
OUTPUT_FILE = Path("/home/arlo/Data/silence_detector/amp_silence_results.json")
BACKUP_FILE = Path("/home/arlo/Data/silence_detector/amp_silence_results.bak.json")

AMP_FPS = 44100 / 4096  # ~10.77 Hz

# Updated threshold
SILENT_FRAME_THRESH = 0.005
MIN_SILENCE_DURATION_SEC = 0.5
GAP_MERGE_SEC = 0.05          # Was 0.3
SIGNIFICANT_SILENCE_SEC = 2.0

MOSTLY_SILENT_RATIO = 0.70
MOSTLY_SILENT_MEAN = 0.05


def find_silent_regions_np(amp, fps):
    """Find contiguous silent regions using vectorized numpy ops."""
    silent_mask = amp < SILENT_FRAME_THRESH
    T = len(silent_mask)
    if T == 0:
        return [], 0.0

    min_frames = max(1, int(MIN_SILENCE_DURATION_SEC * fps))
    gap_frames = max(1, int(GAP_MERGE_SEC * fps))

    padded = np.concatenate(([False], silent_mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return [], 0.0

    if len(starts) > 1:
        gaps = starts[1:] - ends[:-1]
        merge_mask = gaps <= gap_frames
        merged_starts = [starts[0]]
        merged_ends = []
        for i in range(len(merge_mask)):
            if merge_mask[i]:
                continue
            else:
                merged_ends.append(ends[i])
                merged_starts.append(starts[i + 1])
        merged_ends.append(ends[-1])
        starts = np.array(merged_starts)
        ends = np.array(merged_ends)

    durations = ends - starts
    mask = durations >= min_frames
    starts = starts[mask]
    ends = ends[mask]

    regions = []
    total_silent = 0.0
    for s, e in zip(starts, ends):
        start_sec = round(float(s) / fps, 2)
        end_sec = round(float(e) / fps, 2)
        dur = round(end_sec - start_sec, 2)
        total_silent += dur
        regions.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": dur,
        })

    silence_ratio = total_silent / (T / fps) if T > 0 else 0.0
    return regions, round(silence_ratio, 4)


def reclassify(result, regions, silence_ratio):
    old_cls = result["classification"]
    if old_cls in ("fully_silent", "noise_hiss"):
        return old_cls

    mean = result.get("mean", 0)
    if silence_ratio > MOSTLY_SILENT_RATIO and mean < MOSTLY_SILENT_MEAN:
        return "mostly_silent"

    for r in regions:
        if r["duration_sec"] >= SIGNIFICANT_SILENCE_SEC:
            return "has_silent_regions"

    return "normal"


def needs_reprocess(r):
    """Check if this entry was likely over-merged."""
    if r.get("abs_peak", 0) <= 0.01:
        return False  # truly silent, regions are fine
    if r.get("silence_ratio", 0) <= 0.95:
        return False  # regions are reasonably accurate
    return True


def main():
    print(f"Loading {INPUT_FILE}...")
    with open(INPUT_FILE, 'rb') as f:
        data = orjson.loads(f.read())

    results = data.get("results", [])
    print(f"Total results: {len(results):,}")

    # Backup
    if not BACKUP_FILE.exists():
        print(f"Backing up to {BACKUP_FILE}...")
        with open(BACKUP_FILE, 'wb') as f:
            f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    # Find entries needing reprocessing
    to_reprocess = [(i, r) for i, r in enumerate(results) if needs_reprocess(r)]
    print(f"Entries needing reprocessing: {len(to_reprocess):,} / {len(results):,}")

    changed_count = 0
    failed = 0
    reclassified_to_normal = 0

    for idx, r in tqdm(to_reprocess, desc="Recomputing"):
        rel_path = r.get("path", "")
        if not rel_path:
            continue

        stem = os.path.splitext(rel_path)[0]
        amp_path = CONDITIONING_ROOT / f"{stem}.amp.npy"

        try:
            amp = np.load(str(amp_path))
        except Exception:
            failed += 1
            continue

        if len(amp) < 5:
            continue

        old_regions = r.get("silent_regions", [])
        old_cls = r["classification"]

        regions, silence_ratio = find_silent_regions_np(amp, AMP_FPS)
        new_cls = reclassify(r, regions, silence_ratio)

        if old_regions != regions or old_cls != new_cls:
            changed_count += 1

        r["silent_regions"] = regions
        r["num_silent_regions"] = len(regions)
        r["silence_ratio"] = silence_ratio
        r["classification"] = new_cls

    # Count reclassified
    cls_counts = Counter(r["classification"] for r in results)
    reclassified_to_normal = sum(1 for _, r in to_reprocess if r["classification"] == "normal")

    non_normal = [r for r in results if r["classification"] != "normal"]
    non_normal.sort(key=lambda x: -(x.get("silence_ratio", 0)))

    print(f"\nChanged: {changed_count:,} / {len(to_reprocess):,} reprocessed")
    print(f"Failed to load amp: {failed:,}")
    print(f"Reclassified to normal: {reclassified_to_normal:,}")
    print(f"Non-normal remaining: {len(non_normal):,}")
    print("\nClassification breakdown:")
    for cls, count in cls_counts.most_common():
        print(f"  {cls}: {count:,}")

    data["results"] = non_normal
    data["parameters"]["gap_merge_sec"] = GAP_MERGE_SEC
    data["summary"] = dict(cls_counts)
    data["recomputed_at"] = datetime.now().isoformat()
    data["recompute_note"] = f"Recomputed {len(to_reprocess)} over-merged entries with gap_merge_sec={GAP_MERGE_SEC}"

    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'wb') as f:
        f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2))

    print("Done!")


if __name__ == "__main__":
    main()
