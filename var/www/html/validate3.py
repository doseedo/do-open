#!/usr/bin/env python3
"""
Validator + single-bad-element nulling:

Rules hierarchy:
1) Latent is the anchor (must exist & load).
2) Enforce/rescue PR like before (zero-tail crop; silent-tail rescue).
3) Enforce fast-grid for Encodec.
4) Conditioning optional; if mismatched to latent, drop it.

NEW:
- If exactly 1 element in {conditioning, PR, Encodec} is misaligned/fails while the other 2 align,
  replace that element's path with null ({} for conditioning_paths) and accept the entry.
- If 2+ are bad → reject.

Outputs:
- final_training_manifest_final.json (accepted entries only; bad paths nulled)
- validation_errors_final.txt
- validation_valid_specs_final.csv
- validation_actions_final.txt
"""

import json
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
import csv

# =========================
# --- CONFIGURATION ---
# =========================

INPUT_JSON  = Path("final_training_manifest.json")
OUTPUT_JSON = Path("final_training_manifest_final.json")
ERROR_LOG   = Path("validation_errors_final.txt")
VALID_CSV   = Path("validation_valid_specs_final.csv")
ACTIONS_LOG = Path("validation_actions_final.txt")

# Conditioning optional (kept if aligned; dropped if mismatched)
CONDITIONING_OPTIONAL = True

# Optional: write cropped piano-rolls to disk (keeps originals)
PR_CROP_DIR = Path("rescued_pianorolls")
PR_CROP_DIR.mkdir(exist_ok=True, parents=True)

# =========================
# Grid / hop params
# =========================
DCAE_SR = 44100
DCAE_HOP_LENGTH = 4096

ENCODEC_SR = 24000
ENCODEC_HOP_LENGTH = 320  # internal hop

# =========================
# --- TOLERANCES ---
# =========================

# latent vs conditioning (slow grid)
SLOW_SPREAD_ABS_FRAMES = 12
SLOW_SPREAD_REL_FRACTION = 0.06  # ~6%

# PR “minor” shortfall tolerance (accepted without rescue)
PR_SHORT_ABS_FRAMES = 24
PR_SHORT_REL_FRACTION = 0.18

# Encodec fast-grid tolerance
ENC_ABS_FLOOR_FRAMES = 24
ENC_REL_FRACTION = 0.015  # 1.5%

# =========================
# --- RESCUE RULES ---
# =========================

# Define "huge" shortfall → requires silence proof (via conditioning amp)
HUGE_SHORTFALL_ABS_FRAMES = 96
HUGE_SHORTFALL_REL_FRACTION = 0.30

# Silence detection for conditioning amp (rescue PR short)
COND_SILENCE_EPS = 1e-3
COND_SILENCE_FRACTION_QUIET = 0.98  # ≥98% of tail frames <= eps

# Zero padding for PR tails (rescue PR long)
PR_ZERO_EPS = 0.0
PR_ZERO_ALLOW_NEAR = False  # set True & PR_ZERO_EPS small to allow ~zeros instead of strict zeros

# =========================
# Helpers
# =========================

def _within(a: int, b: int, abs_frames: int, rel_frac: float) -> bool:
    diff = abs(a - b)
    return (diff <= abs_frames) or (diff <= rel_frac * max(a, b, 1))

def _safe_np_load(path: Path) -> np.ndarray:
    if not path or str(path).strip() in {"", "."}:
        raise FileNotFoundError("empty path")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        return np.load(path)
    except ValueError:
        # retry with pickle allowed; if still fails, raise
        try:
            return np.load(path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"Corrupt or non-npy file: {path} ({e})")

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

def _extract_encodec_len(obj: Any) -> Optional[int]:
    try:
        t = _first_tensor(obj)
        if t is None:
            return None
        return int(t.shape[-1])
    except Exception:
        return None

def _expected_encodec_len(latent_len: int) -> float:
    slow_rate = DCAE_SR / DCAE_HOP_LENGTH
    fast_rate = ENCODEC_SR / ENCODEC_HOP_LENGTH
    return latent_len * (fast_rate / slow_rate)

def _ok_encodec_len(actual: int, expected: float) -> Tuple[bool, int]:
    tol = max(ENC_ABS_FLOOR_FRAMES, int(ENC_REL_FRACTION * expected))
    return (abs(actual - expected) <= tol, tol)

def _minor_shortfall_ok(pr_len: int, ref_len: int) -> bool:
    if pr_len > ref_len:
        return False
    shortfall = ref_len - pr_len
    return shortfall <= max(PR_SHORT_ABS_FRAMES, int(PR_SHORT_REL_FRACTION * ref_len))

def _huge_shortfall(pr_len: int, ref_len: int) -> bool:
    shortfall = ref_len - pr_len
    return shortfall > max(HUGE_SHORTFALL_ABS_FRAMES, int(HUGE_SHORTFALL_REL_FRACTION * ref_len))

def _cond_tail_is_silent(amp: np.ndarray, start: int, end: int) -> bool:
    if amp is None or amp.ndim == 0:
        return False
    start = max(start, 0)
    end   = min(end, amp.shape[-1])
    if end <= start:
        return False
    tail = amp[..., start:end]
    if tail.size == 0:
        return False
    quiet = (tail <= COND_SILENCE_EPS)
    frac_quiet = float(np.count_nonzero(quiet)) / float(tail.size)
    return frac_quiet >= COND_SILENCE_FRACTION_QUIET

def _pr_tail_all_zero(pr: np.ndarray, start: int) -> bool:
    if pr.ndim < 2:
        return False
    if start >= pr.shape[1]:
        return True
    tail = pr[:, start:]
    if PR_ZERO_ALLOW_NEAR:
        return np.all(np.abs(tail) <= PR_ZERO_EPS)
    return np.all(tail == 0)

def _crop_pr_file(pr_path: Path, new_len: int) -> Path:
    pr = _safe_np_load(pr_path)
    if pr.ndim < 2:
        raise ValueError(f"Piano roll has unexpected shape {pr.shape}.")
    new_len = min(new_len, pr.shape[1])
    cropped = pr[:, :new_len]
    dst = PR_CROP_DIR / (pr_path.stem + f".crop{new_len}" + pr_path.suffix)
    np.save(dst, cropped)
    return dst

# =========================
# Core validation with rescue + single-bad nulling
# =========================

def validate_rescue_and_null(entry: Dict) -> Tuple[bool, str, Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (accepted, message, specs, output_entry)

    Acceptance logic:
    - Try normal validation with PR rescue + conditioning drop + encodec check.
    - If any one of {conditioning, PR, Encodec} fails while the other two pass, null that one and accept.
    - If 2+ fail, reject.
    """
    specs: Dict[str, Any] = {}
    out = dict(entry)
    actions: List[str] = []

    try:
        audio_path = entry.get("audio_path", "<unknown>")
        latent_path = Path(entry.get("latent_path", ""))
        encodec_path = Path(entry.get("encodec_path", ""))
        pr_path = Path(entry.get("piano_roll_path", ""))

        cond_paths = entry.get("conditioning_paths") or {}
        amp_str = cond_paths.get("amp") if isinstance(cond_paths, dict) else None

        specs.update({
            "audio_path": audio_path,
            "latent_path": str(latent_path) if latent_path else "",
            "encodec_path": str(encodec_path) if encodec_path else "",
            "piano_roll_path": str(pr_path) if pr_path else "",
            "conditioning_amp_path": amp_str or "",
            "actions": actions,
        })

        # Required presence
        if not (latent_path and pr_path and encodec_path):
            return (False, "Missing latent/encodec/piano_roll path.", specs, out)
        if not latent_path.exists():
            return (False, "DCAE latent file is missing.", specs, out)

        # Latent length
        latent_blob = torch.load(latent_path, map_location="cpu")
        latent_len = _extract_latent_len(latent_blob)
        if latent_len is None:
            return (False, "Could not read latent length.", specs, out)
        specs["latent_len"] = int(latent_len)

        # Load PR (soft fail allowed later)
        pr_loaded = True
        try:
            pr = _safe_np_load(pr_path)
            if pr.ndim < 2:
                raise RuntimeError(f"PR has bad shape {pr.shape}")
            pr_len = int(pr.shape[1])
        except Exception as e:
            pr_loaded = False
            pr = None
            pr_len = None
            pr_load_err = str(e)

        specs["piano_roll_len"] = pr_len if pr_len is not None else ""

        # Load conditioning amp if present
        cond_present = False
        cond_usable = False
        amp_np = None
        cond_len = None
        if isinstance(amp_str, str) and amp_str.strip():
            ap = Path(amp_str)
            if ap.exists() and ap.is_file():
                try:
                    amp_np = _safe_np_load(ap)
                    cond_len = int(amp_np.shape[-1]) if amp_np.ndim >= 1 else int(len(amp_np))
                    cond_present = True
                except Exception:
                    cond_present = False

        specs["conditioning_present"] = bool(cond_present)
        specs["conditioning_len"] = int(cond_len) if cond_len is not None else ""

        # Check conditioning alignment vs latent
        if cond_present and cond_len is not None:
            lc_ok = _within(latent_len, cond_len, SLOW_SPREAD_ABS_FRAMES, SLOW_SPREAD_REL_FRACTION)
            cond_usable = lc_ok
            specs["lc_ok"] = bool(lc_ok)
            specs["lc_delta"] = int(abs(latent_len - cond_len))
        else:
            specs["lc_ok"] = None
            specs["lc_delta"] = ""

        # Determine ref_len for PR checks
        ref_len = max(latent_len, cond_len) if cond_usable else latent_len
        specs["pr_ref_len"] = int(ref_len)

        # PR validity (with rescue)
        pr_ok = False
        pr_action = None
        if pr_loaded:
            if pr_len < ref_len:
                if _minor_shortfall_ok(pr_len, ref_len):
                    pr_ok = True
                else:
                    if _huge_shortfall(pr_len, ref_len):
                        if cond_usable and _cond_tail_is_silent(amp_np, pr_len, ref_len):
                            pr_ok = True
                            pr_action = "rescue_pr_short_tail_silent"
                        else:
                            pr_ok = False
                    else:
                        if cond_usable and _cond_tail_is_silent(amp_np, pr_len, ref_len):
                            pr_ok = True
                            pr_action = "rescue_pr_short_tail_silent"
                        else:
                            pr_ok = False
            elif pr_len == ref_len:
                pr_ok = True
            else:
                # PR longer than ref: accept only if tail zeros → crop
                if _pr_tail_all_zero(pr, ref_len):
                    try:
                        new_pr_path = _crop_pr_file(pr_path, ref_len)
                        out = dict(out)
                        out["piano_roll_path"] = str(new_pr_path)
                        pr_ok = True
                        pr_action = f"crop_pr_tail_to_{ref_len}"
                        specs["piano_roll_len_after_crop"] = ref_len
                    except Exception as e:
                        pr_ok = False
                        pr_action = f"crop_failed:{e}"
                else:
                    pr_ok = False
        else:
            pr_ok = False  # not loaded → treated as fail

        specs["pr_ok"] = bool(pr_ok)
        if pr_action:
            actions.append(pr_action)

        # Encodec validity
        enc_ok = False
        try:
            enc_blob = torch.load(encodec_path, map_location="cpu")
            encodec_len = _extract_encodec_len(enc_blob)
            if encodec_len is None:
                enc_ok = False
                encodec_len = ""
            else:
                exp = _expected_encodec_len(latent_len)
                enc_ok, tol = _ok_encodec_len(encodec_len, exp)
                specs.update({
                    "encodec_len": int(encodec_len),
                    "encodec_expected": int(round(exp)),
                    "encodec_delta": int(abs(int(encodec_len) - int(round(exp)))),
                    "encodec_tol_abs": int(tol),
                })
        except Exception as e:
            enc_ok = False
            specs["encodec_len"] = ""
            specs["encodec_expected"] = ""
            specs["encodec_delta"] = f"load_err:{e}"

        specs["enc_ok"] = bool(enc_ok)

        # Conditioning acceptance: optional; if mismatched, drop
        cond_ok = True  # treating "absent" as OK
        if cond_present:
            if cond_usable:
                cond_ok = True
            else:
                # drop from output
                out = dict(out)
                out["conditioning_paths"] = {}
                actions.append("drop_conditioning_mismatch")
                cond_ok = False  # mark as a "fail" for the single-bad-element rule

        # -----------------------------
        # Decide acceptance with single-bad-element nulling
        # -----------------------------
        # Count how many of {PR, Encodec, Conditioning} failed (False)
        # Note: cond_ok False only when present and mismatched; absent counts as OK.
        fail_list = []
        if not pr_ok:
            fail_list.append("pr")
        if not enc_ok:
            fail_list.append("encodec")
        if not cond_ok:
            fail_list.append("conditioning")

        if len(fail_list) == 0:
            actions.append("accept_normal")
            specs["actions"] = actions
            return (True, "Valid", specs, out)

        if len(fail_list) == 1:
            bad = fail_list[0]
            # Null the bad element’s path and accept
            out = dict(out)
            if bad == "pr":
                out["piano_roll_path"] = None
                actions.append("null_pr_path_accept")
            elif bad == "encodec":
                out["encodec_path"] = None
                actions.append("null_encodec_path_accept")
            elif bad == "conditioning":
                # already dropped to {}; still accept
                actions.append("null_conditioning_accept")
            specs["actions"] = actions
            return (True, f"Accepted with single-bad-element nulled: {bad}", specs, out)

        # 2+ failed → reject
        reason_bits = []
        if not pr_ok:
            if not pr_loaded:
                reason_bits.append(f"PR load error: {pr_load_err}")
            else:
                reason_bits.append("PR misaligned")
        if not enc_ok:
            reason_bits.append("Encodec misaligned")
        if not cond_ok:
            reason_bits.append("Conditioning mismatched")
        reason = "; ".join(reason_bits) if reason_bits else "Multiple elements failed"
        return (False, reason, specs, out)

    except Exception as e:
        return (False, f"Error processing entry: {e}", specs, out)

# =========================
# Main
# =========================

def main():
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    with INPUT_JSON.open("r") as f:
        data = json.load(f)

    print(f"🔬 Validating {len(data)} entries from {INPUT_JSON} with single-bad nulling...")

    val_specs: List[Dict[str, Any]] = []
    invalid_logs: List[str] = []
    actions_log: List[str] = []
    accepted: List[Dict[str, Any]] = []

    for entry in tqdm(data, desc="Checking"):
        ok, msg, specs, out_entry = validate_rescue_and_null(entry)
        if ok:
            specs["group"] = entry.get("group", "")
            specs["sub_group"] = entry.get("sub_group", "")
            val_specs.append(specs)
            accepted.append(out_entry)
            if specs.get("actions"):
                actions_log.append(f"{specs['audio_path']}: " + ", ".join(specs["actions"]))
        else:
            audio_path = entry.get("audio_path", "<unknown>")
            invalid_logs.append(f"File: {audio_path}\nReason: {msg}\n")

    # Report
    print("\n--- Final Report ---")
    print(f"Total Entries Checked: {len(data):,}")
    print(f"✅ Accepted Entries:   {len(accepted):,}")
    print(f"❌ Rejected Entries:   {len(invalid_logs):,}")
    print("---------------------\n")

    # Write outputs
    with OUTPUT_JSON.open("w") as f:
        json.dump(accepted, f, indent=2)
    print(f"🧾 Final manifest saved to: {OUTPUT_JSON.resolve()}")

    if val_specs:
        fieldnames = [
            "audio_path", "group", "sub_group",
            "latent_path", "piano_roll_path", "encodec_path", "conditioning_amp_path",
            "latent_len", "conditioning_len", "piano_roll_len",
            "pr_ref_len", "pr_ok", "enc_ok", "lc_ok", "lc_delta",
            "encodec_expected", "encodec_len", "encodec_delta", "encodec_tol_abs",
        ]
        try:
            with VALID_CSV.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                for row in val_specs:
                    out = {k: row.get(k, "") for k in fieldnames}
                    w.writerow(out)
            print(f"✅ Valid specs CSV saved to: {VALID_CSV.resolve()}")
        except Exception as e:
            print(f"⚠️ Could not write CSV: {e}")

    if invalid_logs:
        with ERROR_LOG.open("w") as f:
            f.write("--- Rejected entries ---\n\n")
            f.write("\n".join(invalid_logs))
        print(f"❌ Errors saved to: {ERROR_LOG.resolve()}")

    if actions_log:
        with ACTIONS_LOG.open("w") as f:
            f.write("--- Actions taken per accepted entry ---\n")
            f.write("\n".join(actions_log))
        print(f"📝 Actions log saved to: {ACTIONS_LOG.resolve()}")

if __name__ == "__main__":
    main()
