#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
import csv

# --- CONFIGURATION ---

INPUT_JSON  = Path("final_training_manifest.json")
ERROR_LOG   = Path("validation_errors.txt")
VALID_CSV   = Path("validation_valid_specs.csv")
OUTPUT_JSON = Path("final_training_manifest.final.json")   # NEW: final trimmed+aligned manifest

# Conditioning optional: if True, null/missing conditioning_paths WILL NOT invalidate an entry
CONDITIONING_OPTIONAL = True

# Grid / hop params
DCAE_SR = 44100
DCAE_HOP_LENGTH = 4096

ENCODEC_SR = 24000
ENCODEC_HOP_LENGTH = 320  # internal hop

# --- TOLERANCES ---

# "Slow grid" (latent / conditioning / piano roll) tolerances
# latent & conditioning tight match (used only when conditioning is present)
SLOW_SPREAD_ABS_FRAMES = 12
SLOW_SPREAD_REL_FRACTION = 0.06  # ~6%

# Piano roll can be SHORTER than reference (content stops), but not longer
PR_SHORT_ABS_FRAMES = 12
PR_SHORT_REL_FRACTION = 0.15

# "Fast grid" (Encodec) tolerance
ENC_ABS_FLOOR_FRAMES = 18
ENC_REL_FRACTION = 0.01  # 1% of expected

# ---------------------------------------------------------------------------

def _canon_path(p: str) -> str:
    """Canonicalize path strings (strip, collapse //, normpath)."""
    if not isinstance(p, str):
        return p
    s = p.strip()
    while "//" in s:
        s = s.replace("//", "/")
    return os.path.normpath(s)

def _within(a: int, b: int, abs_frames: int, rel_frac: float) -> bool:
    diff = abs(a - b)
    return (diff <= abs_frames) or (diff <= rel_frac * max(a, b, 1))

def _piano_roll_short_ok(pr_len: int, ref_len: int) -> bool:
    if pr_len > ref_len:
        return False
    shortfall = ref_len - pr_len
    return shortfall <= max(PR_SHORT_ABS_FRAMES, int(PR_SHORT_REL_FRACTION * ref_len))

def _extract_latent_len(latent_blob: Any) -> Optional[int]:
    try:
        if isinstance(latent_blob, dict) and "latents" in latent_blob:
            t = latent_blob["latents"]
            if isinstance(t, torch.Tensor):
                return int(t.shape[-1])
        if isinstance(latent_blob, torch.Tensor):
            return int(latent_blob.shape[-1])
    except Exception:
        pass
    return None

def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for k in ("codes", "tokens", "encodec", "audio_tokens", "data"):
            if k in x:
                t = _first_tensor(x[k])
                if t is not None:
                    return t
        for v in x.values():
            t = _first_tensor(v)
            if t is not None:
                return t
    return None

def _extract_encodec_len(obj: Any) -> Optional[int]:
    try:
        t = _first_tensor(obj)
        if t is None:
            return None
        return int(t.shape[-1])
    except Exception:
        return None

def _safe_np_load(path: Path) -> np.ndarray:
    # Only load real files; never try to load "" or "."
    if not path or str(path).strip() in {"", "."}:
        raise FileNotFoundError("empty path")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        return np.load(path)
    except ValueError:
        return np.load(path, allow_pickle=True)

def _expected_encodec_len(latent_len: int) -> float:
    slow_rate = DCAE_SR / DCAE_HOP_LENGTH
    fast_rate = ENCODEC_SR / ENCODEC_HOP_LENGTH
    return latent_len * (fast_rate / slow_rate)

def _ok_encodec_len(actual: int, expected: float) -> bool:
    tol = max(ENC_ABS_FLOOR_FRAMES, int(ENC_REL_FRACTION * expected))
    return abs(actual - expected) <= tol

def validate_entry(entry: Dict) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns: (is_valid, reason, specs_dict)
    specs_dict includes lengths and final trimmed targets if valid.
    """
    specs: Dict[str, Any] = {}
    try:
        audio_path = _canon_path(entry.get("audio_path", "<unknown>"))
        latent_path = Path(_canon_path(entry.get("latent_path", "")))
        encodec_path = Path(_canon_path(entry.get("encodec_path", "")))
        pr_path = Path(_canon_path(entry.get("piano_roll_path", "")))

        # Conditioning presence (optional)
        cond_paths = entry.get("conditioning_paths") or {}
        amp_str = cond_paths.get("amp") if isinstance(cond_paths, dict) else None
        amp_path = Path(_canon_path(amp_str)) if isinstance(amp_str, str) and amp_str.strip() else None

        specs.update({
            "audio_path": audio_path,
            "latent_path": str(latent_path) if latent_path else "",
            "encodec_path": str(encodec_path) if encodec_path else "",
            "piano_roll_path": str(pr_path) if pr_path else "",
            "conditioning_amp_path": str(amp_path) if amp_path else "",
            "conditioning_present": False,
        })

        # Required path keys presence (conditioning can be optional)
        if not (latent_path and encodec_path and pr_path):
            return (False, "Entry missing latent, encodec, or piano_roll path.", specs)

        # Check existence of required files
        if not latent_path.exists():
            return (False, "DCAE latent file is missing.", specs)
        if not encodec_path.exists():
            return (False, "Encodec token file is missing.", specs)
        if not pr_path.exists():
            return (False, "Piano roll file is missing.", specs)

        # Conditioning availability
        conditioning_available = False
        if amp_path and amp_path.exists() and amp_path.is_file():
            conditioning_available = True
        elif not CONDITIONING_OPTIONAL:
            return (False, "Conditioning required but missing.", specs)

        specs["conditioning_present"] = conditioning_available

        # --- Load lengths ---
        latent_blob = torch.load(latent_path, map_location="cpu")
        latent_len = _extract_latent_len(latent_blob)
        if latent_len is None:
            return (False, "Could not read latent length.", specs)
        latent_len = int(latent_len)
        specs["latent_len"] = latent_len

        pr_data = _safe_np_load(pr_path)
        if pr_data.ndim < 2:
            return (False, f"Piano roll has unexpected shape {pr_data.shape}.", specs)
        piano_roll_len = int(pr_data.shape[1])
        specs["piano_roll_len"] = piano_roll_len

        conditioning_len: Optional[int] = None
        if conditioning_available and amp_path is not None:
            amp_data = _safe_np_load(amp_path)
            conditioning_len = int(amp_data.shape[-1]) if amp_data.ndim >= 1 else int(len(amp_data))
            specs["conditioning_len"] = conditioning_len
        else:
            specs["conditioning_len"] = None

        # --- Slow grid checks (as before) ---
        if conditioning_len is not None:
            lc_ok = _within(latent_len, conditioning_len, SLOW_SPREAD_ABS_FRAMES, SLOW_SPREAD_REL_FRACTION)
            specs["lc_ok"] = bool(lc_ok)
            specs["lc_delta"] = int(abs(latent_len - conditioning_len))
            specs["lc_tol_abs"] = int(SLOW_SPREAD_ABS_FRAMES)
            specs["lc_tol_rel_pct"] = int(SLOW_SPREAD_REL_FRACTION * 100)
            if not lc_ok:
                return (False,
                        f"Slow Grid Alignment Error: Latent ({latent_len}), Conditioning ({conditioning_len}) "
                        f"do not match within spread ≤{SLOW_SPREAD_ABS_FRAMES} or ≤{int(SLOW_SPREAD_REL_FRACTION*100)}%.",
                        specs)
            ref_len = max(latent_len, conditioning_len)
        else:
            specs["lc_ok"] = None
            specs["lc_delta"] = None
            specs["lc_tol_abs"] = int(SLOW_SPREAD_ABS_FRAMES)
            specs["lc_tol_rel_pct"] = int(SLOW_SPREAD_REL_FRACTION * 100)
            ref_len = latent_len

        pr_ok = _piano_roll_short_ok(piano_roll_len, ref_len)
        specs["pr_ok"] = bool(pr_ok)
        specs["pr_ref_len"] = int(ref_len)
        specs["pr_shortfall"] = int(ref_len - piano_roll_len)
        specs["pr_short_allow_abs"] = int(PR_SHORT_ABS_FRAMES)
        specs["pr_short_allow_rel_pct"] = int(PR_SHORT_REL_FRACTION * 100)
        if not pr_ok:
            return (False,
                    f"Slow Grid Alignment Error: Piano Roll ({piano_roll_len}) not acceptable vs ref ({ref_len}); "
                    f"it may be shorter by ≤max({PR_SHORT_ABS_FRAMES}, {int(PR_SHORT_REL_FRACTION*100)}%).",
                    specs)

        # --- Fast grid (Encodec) check ---
        enc_blob = torch.load(encodec_path, map_location="cpu")
        encodec_len = _extract_encodec_len(enc_blob)
        if encodec_len is None:
            return (False, "Could not read Encodec token length.", specs)
        encodec_len = int(encodec_len)
        specs["encodec_len"] = encodec_len

        expected_enc = _expected_encodec_len(latent_len)
        enc_ok = _ok_encodec_len(encodec_len, expected_enc)
        tol = max(ENC_ABS_FLOOR_FRAMES, int(ENC_REL_FRACTION * expected_enc))
        specs["encodec_expected"] = int(round(expected_enc))
        specs["encodec_tol_abs"] = int(tol)
        specs["encodec_delta"] = int(abs(int(encodec_len) - int(round(expected_enc))))
        specs["enc_ok"] = bool(enc_ok)
        if not enc_ok:
            return (False,
                    f"Fast Grid Alignment Error: Encodec length ({encodec_len}) not within ±{tol} of "
                    f"expected ~{int(round(expected_enc))} from latent ({latent_len}).",
                    specs)

        # --- NEW: Compute final trimmed targets for training ---
        # Trim slow grid to the MIN across available modalities so they align perfectly.
        slow_trim_target = latent_len
        if conditioning_len is not None:
            slow_trim_target = min(slow_trim_target, conditioning_len)
        slow_trim_target = min(slow_trim_target, piano_roll_len)
        slow_trim_target = max(int(slow_trim_target), 1)  # safety

        # Encodec target = expected from slow_trim_target, but never exceed file length
        enc_expected_from_trim = int(round(_expected_encodec_len(slow_trim_target)))
        fast_trim_target = min(encodec_len, enc_expected_from_trim)
        fast_trim_target = max(int(fast_trim_target), 1)

        specs["final_slow_len"] = int(slow_trim_target)
        specs["final_fast_len"] = int(fast_trim_target)

        # If we reached here, it's valid
        return (True, "Valid", specs)

    except Exception as e:
        return (False, f"Error processing entry: {e}", specs)

def _build_final_entry(orig: Dict[str, Any], specs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the final JSON entry the PyTorch dataset can ingest directly.
    - canonicalized paths
    - trimmed frame targets
    - meta (sr/hops) and offsets (0 for head-aligned, tail-trimmed)
    """
    # Canonicalize paths
    audio_path   = _canon_path(orig.get("audio_path", ""))
    latent_path  = _canon_path(orig.get("latent_path", ""))
    encodec_path = _canon_path(orig.get("encodec_path", ""))
    pr_path      = _canon_path(orig.get("piano_roll_path", ""))

    cond_paths = {}
    if isinstance(orig.get("conditioning_paths"), dict):
        for k, v in orig["conditioning_paths"].items():
            cond_paths[k] = _canon_path(v) if isinstance(v, str) else v

    final_entry = {
        "audio_path": audio_path,
        "latent_path": latent_path,
        "encodec_path": encodec_path,
        "piano_roll_path": pr_path,
        "conditioning_paths": cond_paths,
        "group": orig.get("group", ""),
        "sub_group": orig.get("sub_group", ""),
        "frames": {
            "slow": int(specs["final_slow_len"]),   # frames at DCAE_HOP_LENGTH
            "fast": int(specs["final_fast_len"]),   # frames at ENCODEC_HOP_LENGTH
        },
        "offsets": {
            "slow": 0,   # head-aligned; dataset should read [0:slow]
            "fast": 0,
            "piano_roll": 0,
            "conditioning": 0,
        },
        "meta": {
            "slow_sr": DCAE_SR,
            "slow_hop": DCAE_HOP_LENGTH,
            "fast_sr": ENCODEC_SR,
            "fast_hop": ENCODEC_HOP_LENGTH,
            "has_conditioning": bool(specs.get("conditioning_present", False)),
        },
    }
    return final_entry

def main():
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    with INPUT_JSON.open("r") as f:
        data = json.load(f)

    print(f"🔬 Validating {len(data)} entries from {INPUT_JSON}...")

    valid_specs: List[Dict[str, Any]] = []
    valid_pairs: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    invalid_logs: List[str] = []

    for entry in tqdm(data, desc="Validating Dataset"):
        ok, reason, specs = validate_entry(entry)
        if ok:
            # add a few helpful identity fields if present
            specs["group"] = entry.get("group", "")
            specs["sub_group"] = entry.get("sub_group", "")
            valid_specs.append(specs)
            valid_pairs.append((entry, specs))
        else:
            audio_path = entry.get("audio_path", "<unknown>")
            invalid_logs.append(f"File: {audio_path}\nReason: {reason}\n")

    # --- Report ---
    print("\n--- Validation Report ---")
    print(f"Total Entries Checked: {len(data):,}")
    print(f"✅ Valid Entries:       {len(valid_specs):,}")
    print(f"❌ Invalid Entries:     {len(invalid_logs):,}")
    print("-------------------------\n")

    # Write invalid text log (unchanged behavior)
    if invalid_logs:
        with ERROR_LOG.open("w") as f:
            f.write("--- Dataset Validation Errors ---\n\n")
            f.write("\n".join(invalid_logs))
        print(f"Detailed invalid report saved to: {ERROR_LOG.resolve()}")
    else:
        print("🎉 All entries are valid under the current tolerances!")

    # Write valid specs CSV (unchanged)
    if valid_specs:
        fieldnames = [
            "audio_path",
            "group", "sub_group",
            "latent_path", "piano_roll_path", "encodec_path", "conditioning_amp_path",
            "conditioning_present",
            "latent_len", "conditioning_len", "piano_roll_len",
            "lc_ok", "lc_delta", "lc_tol_abs", "lc_tol_rel_pct",
            "pr_ok", "pr_ref_len", "pr_shortfall", "pr_short_allow_abs", "pr_short_allow_rel_pct",
            "encodec_len", "encodec_expected", "encodec_delta", "encodec_tol_abs", "enc_ok",
            "final_slow_len", "final_fast_len",   # NEW
        ]
        try:
            with VALID_CSV.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in valid_specs:
                    out = {k: row.get(k, "") for k in fieldnames}
                    writer.writerow(out)
            print(f"✅ Valid specs CSV saved to: {VALID_CSV.resolve()}")
        except Exception as e:
            print(f"⚠️ Could not write valid specs CSV: {e}")

    # --- Build and write FINAL dataset manifest (trimmed & aligned) ---
    final_entries: List[Dict[str, Any]] = []
    for orig, specs in valid_pairs:
        final_entries.append(_build_final_entry(orig, specs))

    OUTPUT_JSON.write_text(json.dumps(final_entries, indent=2))
    print(f"✅ Final trimmed+aligned manifest saved to: {OUTPUT_JSON.resolve()}")

if __name__ == "__main__":
    main()
