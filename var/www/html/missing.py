#!/usr/bin/env python3
"""
Find conditioning filename mismatches vs audio slice stem and log the original audio_path.

A "mismatch" is when the basename of a conditioning file != "<audio_stem><suffix>".
Example: audio "Bass Body.01_02.wav" but cond "Bass Body.onsets.npy" -> UNSLICED.

Outputs:
- mismatch_logs/mismatched_audio_paths.txt   (unique audio_path per entry with any mismatch)
- mismatch_logs/mismatches_detailed.csv      (one row per mismatch with reason)

Usage:
  python log_slice_mismatches.py \
      --input final_training_manifest.json \
      --log-dir mismatch_logs
"""

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict
from tqdm import tqdm

DEFAULT_INPUT = "final_training_manifest.json"
DEFAULT_LOGDIR = "mismatch_logs"

# Standard conditioning suffixes
COND_SUFFIX: Dict[str, str] = {
    "onsets":    ".onsets.npy",
    "rframe":    ".rframe.npy",
    "rbend":     ".rbend.npy",
    "amp":       ".amp.npy",
    "f0":        ".f0.npy",
    "f0_masked": ".f0_masked.npy",
}

SLICE_RE = re.compile(r"\.\d+_\d+$")  # matches trailing ".01_02"

def norm_str(s: str) -> str:
    s = s.strip()
    while "//" in s:
        s = s.replace("//", "/")
    return os.path.normpath(s)

def stem_base_and_sliced(stem: str) -> Tuple[str, bool]:
    m = SLICE_RE.search(stem)
    if m:
        return stem[:m.start()], True
    return stem, False

def detect_reason(cur_base: str, expected_name: str, base: str, suffix: str) -> str:
    """
    Classify why the basename is mismatched.
    """
    if cur_base == expected_name:
        return ""  # no mismatch
    if cur_base == f"{base}{suffix}":
        return "UNSLICED"  # missing slice like ".01_02"
    # Different slice numbers or extra/other tokens but same suffix
    if cur_base.endswith(suffix) and cur_base.startswith(base + "."):
        return "DIFFERENT_SLICE"
    return "BASENAME_MISMATCH"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Log conditioning filename mismatches vs audio slice.")
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input manifest JSON (list of entries).")
    p.add_argument("--log-dir", default=DEFAULT_LOGDIR, help="Directory to write logs.")
    p.add_argument("--use-entry-cond-keys", action="store_true",
                   help="Validate whatever keys appear per entry's conditioning_paths instead of the default set.")
    return p.parse_args()

def main():
    args = parse_args()
    in_path = Path(args.input)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    if not in_path.exists():
        print(f"❌ Manifest not found: {in_path}")
        return

    try:
        data = json.loads(in_path.read_text())
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return

    if not isinstance(data, list):
        print("❌ Manifest root must be a JSON list of entries.")
        return

    mismatched_audio_paths = set()
    detailed_rows: List[Dict[str, Any]] = []

    print(f"Scanning {len(data):,} entries for filename mismatches…")
    for i, entry in enumerate(tqdm(data, desc="Checking")):
        audio_path = entry.get("audio_path") or ""
        if isinstance(audio_path, str):
            audio_path = norm_str(audio_path)
        stem = Path(audio_path).stem
        base, is_sliced = stem_base_and_sliced(stem)

        cond = entry.get("conditioning_paths") or {}
        if not isinstance(cond, dict):
            # Skip if malformed
            continue

        cond_keys = list(cond.keys()) if args.use_entry_cond_keys else list(COND_SUFFIX.keys())

        entry_had_mismatch = False
        for key in cond_keys:
            suffix = COND_SUFFIX.get(key)
            if not suffix:
                # unknown key; skip
                continue

            cur = cond.get(key)
            if not isinstance(cur, str) or not cur.strip():
                # Missing path entirely — still a mismatch worth logging
                detailed_rows.append({
                    "entry_index": i,
                    "audio_path": audio_path,
                    "key": key,
                    "cond_path": "",
                    "cond_basename": "",
                    "expected_basename": f"{stem}{suffix}",
                    "reason": "MISSING_PATH",
                })
                entry_had_mismatch = True
                continue

            cur_norm = norm_str(cur)
            cur_base = Path(cur_norm).name
            expected_name = f"{stem}{suffix}"
            reason = detect_reason(cur_base, expected_name, base, suffix)

            if reason:
                detailed_rows.append({
                    "entry_index": i,
                    "audio_path": audio_path,
                    "key": key,
                    "cond_path": cur_norm,
                    "cond_basename": cur_base,
                    "expected_basename": expected_name,
                    "reason": reason,
                })
                entry_had_mismatch = True

        if entry_had_mismatch:
            mismatched_audio_paths.add(audio_path)

    # Write unique audio paths
    (log_dir / "mismatched_audio_paths.txt").write_text(
        "\n".join(sorted(mismatched_audio_paths)) + ("\n" if mismatched_audio_paths else "")
    )

    # Write detailed CSV
    if detailed_rows:
        csv_path = log_dir / "mismatches_detailed.csv"
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(detailed_rows[0].keys()))
            w.writeheader()
            w.writerows(detailed_rows)

    # Summary
    print("\n--- Mismatch Report ---")
    print(f"Entries scanned:                {len(data):,}")
    print(f"Entries w/ any mismatch:        {len(mismatched_audio_paths):,}")
    print(f"Unique audio paths logged:      {len(mismatched_audio_paths):,}")
    if detailed_rows:
        print(f"Detailed CSV:                   {(log_dir / 'mismatches_detailed.csv').resolve()}")
    print(f"Audio paths list:               {(log_dir / 'mismatched_audio_paths.txt').resolve()}")

if __name__ == "__main__":
    main()
