#!/usr/bin/env python3
import json
from pathlib import Path
from typing import Tuple, Optional, Any, Dict, List

import numpy as np
import torch
from tqdm import tqdm
import csv

# --- CONFIGURATION ---

INPUT_JSON = Path("vocal_training_manifest.json")
ERROR_LOG  = Path("vocal_validation_errors.txt")
VALID_CSV  = Path("vocal_validation_valid_specs.csv")

# Grid / hop params
DCAE_SR = 44100
DCAE_HOP_LENGTH = 4096

ENCODEC_SR = 24000
ENCODEC_HOP_LENGTH = 320

# --- TOLERANCES ---

# "Slow grid" (latent / conditioning / piano roll) tolerances
SLOW_SPREAD_ABS_FRAMES = 12
SLOW_SPREAD_REL_FRACTION = 0.06  # ~6%

# Piano roll can be SHORTER than reference (content stops), but not longer
PR_SHORT_ABS_FRAMES = 24
PR_SHORT_REL_FRACTION = 0.18

# "Fast grid" (Encodec) tolerance
ENC_ABS_FLOOR_FRAMES = 24
ENC_REL_FRACTION = 0.015  # 1.5% of expected

# ---------------------------------------------------------------------------

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
    specs_dict is populated for both valid and invalid entries to help debugging.
    """
    specs: Dict[str, Any] = {}
    try:
        audio_path = entry.get("audio_path", "<unknown>")
        latent_path = Path(entry.get("latent_path", ""))
        encodec_path = Path(entry.get("encodec_path", ""))
        pr_path = Path(entry.get("piano_roll_path", ""))

        # Standard conditioning
        cond_paths = entry.get("conditioning_paths") or {}
        amp_str = cond_paths.get("amp") if isinstance(cond_paths, dict) else None

        # Vocal conditioning (NEW)
        vocal_cond_paths = entry.get("vocal_conditioning_paths") or {}
        lyrics_data_str = vocal_cond_paths.get("lyrics_data") if isinstance(vocal_cond_paths, dict) else None
        lyrics_tensors_str = vocal_cond_paths.get("lyrics_tensors") if isinstance(vocal_cond_paths, dict) else None
        syllable_boundaries_str = vocal_cond_paths.get("syllable_boundaries") if isinstance(vocal_cond_paths, dict) else None

        specs.update({
            "audio_path": audio_path,
            "latent_path": str(latent_path) if latent_path else "",
            "encodec_path": str(encodec_path) if encodec_path else "",
            "piano_roll_path": str(pr_path) if pr_path else "",
            "conditioning_amp_path": amp_str or "",
            "conditioning_present": False,
            "vocal_lyrics_data_path": lyrics_data_str or "",
            "vocal_lyrics_tensors_path": lyrics_tensors_str or "",
            "vocal_syllable_boundaries_path": syllable_boundaries_str or "",
            "vocal_conditioning_present": False,
        })

        # Required path keys presence
        if not (latent_path and encodec_path and pr_path):
            return (False, "Entry missing latent, encodec, or piano_roll path.", specs)

        # Check existence of required files
        if not latent_path.exists():
            return (False, "DCAE latent file is missing.", specs)
        if not encodec_path.exists():
            return (False, "Encodec token file is missing.", specs)
        if not pr_path.exists():
            return (False, "Piano roll file is missing.", specs)

        # Standard conditioning availability
        conditioning_available = False
        amp_path: Optional[Path] = None
        if isinstance(amp_str, str) and amp_str.strip():
            ap = Path(amp_str)
            if ap.exists() and ap.is_file():
                amp_path = ap
                conditioning_available = True
        specs["conditioning_present"] = conditioning_available

        # Vocal conditioning availability (NEW)
        vocal_conditioning_available = False
        lyrics_data_path: Optional[Path] = None
        lyrics_tensors_path: Optional[Path] = None
        syllable_boundaries_path: Optional[Path] = None

        # Check all vocal conditioning files exist
        if (isinstance(lyrics_data_str, str) and lyrics_data_str.strip() and
            isinstance(lyrics_tensors_str, str) and lyrics_tensors_str.strip() and
            isinstance(syllable_boundaries_str, str) and syllable_boundaries_str.strip()):

            ld_path = Path(lyrics_data_str)
            lt_path = Path(lyrics_tensors_str)
            sb_path = Path(syllable_boundaries_str)

            if not ld_path.exists():
                return (False, f"Vocal conditioning: lyrics_data file missing: {ld_path}", specs)
            if not lt_path.exists():
                return (False, f"Vocal conditioning: lyrics_tensors file missing: {lt_path}", specs)
            if not sb_path.exists():
                return (False, f"Vocal conditioning: syllable_boundaries file missing: {sb_path}", specs)

            lyrics_data_path = ld_path
            lyrics_tensors_path = lt_path
            syllable_boundaries_path = sb_path
            vocal_conditioning_available = True
        else:
            return (False, "Vocal conditioning paths incomplete or missing.", specs)

        specs["vocal_conditioning_present"] = vocal_conditioning_available

        # --- Load lengths ---
        latent_blob = torch.load(latent_path, map_location="cpu")
        latent_len = _extract_latent_len(latent_blob)
        if latent_len is None:
            return (False, "Could not read latent length.", specs)
        specs["latent_len"] = int(latent_len)

        pr_data = _safe_np_load(pr_path)
        if pr_data.ndim < 2:
            return (False, f"Piano roll has unexpected shape {pr_data.shape}.", specs)
        piano_roll_len = int(pr_data.shape[1])
        specs["piano_roll_len"] = piano_roll_len

        conditioning_len: Optional[int] = None
        if conditioning_available and amp_path is not None:
            amp_data = _safe_np_load(amp_path)
            conditioning_len = int(amp_data.shape[-1]) if amp_data.ndim >= 1 else int(len(amp_data))
            specs["conditioning_len"] = int(conditioning_len)
        else:
            specs["conditioning_len"] = None

        # --- Vocal conditioning length checks (NEW) ---
        if vocal_conditioning_available:
            # Check syllable boundaries length
            syllable_data = _safe_np_load(syllable_boundaries_path)
            syllable_len = int(syllable_data.shape[-1]) if syllable_data.ndim >= 1 else int(len(syllable_data))
            specs["syllable_boundaries_len"] = int(syllable_len)

            # Check if syllable boundaries match latent length
            syll_ok = _within(latent_len, syllable_len, SLOW_SPREAD_ABS_FRAMES, SLOW_SPREAD_REL_FRACTION)
            specs["syllable_ok"] = bool(syll_ok)
            specs["syllable_delta"] = int(abs(latent_len - syllable_len))
            if not syll_ok:
                return (False,
                        f"Vocal Conditioning Alignment Error: Syllable boundaries ({syllable_len}) do not match "
                        f"latent ({latent_len}) within spread ≤{SLOW_SPREAD_ABS_FRAMES} or ≤{int(SLOW_SPREAD_REL_FRACTION*100)}%.",
                        specs)

            # Validate lyrics_data JSON structure
            try:
                with lyrics_data_path.open("r") as f:
                    lyrics_json = json.load(f)
                if "syllable_timings" not in lyrics_json:
                    return (False, "Vocal conditioning: lyrics_data JSON missing 'syllable_timings'.", specs)
                specs["lyrics_syllable_count"] = len(lyrics_json.get("syllable_timings", []))
            except Exception as e:
                return (False, f"Vocal conditioning: Error loading lyrics_data JSON: {e}", specs)

            # Validate lyrics_tensors
            try:
                lyrics_tensors_blob = torch.load(lyrics_tensors_path, map_location="cpu")
                if not isinstance(lyrics_tensors_blob, dict):
                    return (False, "Vocal conditioning: lyrics_tensors should be a dict.", specs)
                # Check for expected keys
                expected_keys = {"lyrics_indices", "lyrics_embeddings", "phoneme_embeddings"}
                missing_keys = expected_keys - set(lyrics_tensors_blob.keys())
                if missing_keys:
                    return (False, f"Vocal conditioning: lyrics_tensors missing keys: {missing_keys}", specs)
                specs["lyrics_tensors_keys"] = list(lyrics_tensors_blob.keys())
            except Exception as e:
                return (False, f"Vocal conditioning: Error loading lyrics_tensors: {e}", specs)
        else:
            specs["syllable_boundaries_len"] = None
            specs["syllable_ok"] = None
            specs["syllable_delta"] = None
            specs["lyrics_syllable_count"] = None
            specs["lyrics_tensors_keys"] = None

        # --- Slow grid checks ---
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
        specs["encodec_len"] = int(encodec_len)

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

        # If we reached here, it's valid
        return (True, "Valid", specs)

    except Exception as e:
        return (False, f"Error processing entry: {e}", specs)

def main():
    if not INPUT_JSON.exists():
        print(f"Error: Input JSON file not found at {INPUT_JSON}")
        return

    with INPUT_JSON.open("r") as f:
        data = json.load(f)

    print(f"🔬 Validating {len(data)} vocal entries from {INPUT_JSON}...")

    valid_specs: List[Dict[str, Any]] = []
    invalid_logs: List[str] = []

    for entry in tqdm(data, desc="Validating Vocal Dataset"):
        ok, reason, specs = validate_entry(entry)
        if ok:
            specs["group"] = entry.get("group", "")
            specs["sub_group"] = entry.get("sub_group", "")
            valid_specs.append(specs)
        else:
            audio_path = entry.get("audio_path", "<unknown>")
            invalid_logs.append(f"File: {audio_path}\nReason: {reason}\n")

    # Check for outliers and NaN values in latents
    print("\n🔍 Checking for latent outliers and NaN values...")
    for entry in tqdm(data, desc="Checking Latent Quality"):
        try:
            latent = torch.load(entry["latent_path"], map_location="cpu")
            latent_tensor = latent["latents"] if isinstance(latent, dict) and "latents" in latent else latent
            if not isinstance(latent_tensor, torch.Tensor):
                print(f"ERROR: {entry['audio_path']} has invalid latent format (not a tensor)")
                continue
            if latent_tensor.abs().max() > 20:
                print(f"OUTLIER: {entry['audio_path']} has max latent value {latent_tensor.abs().max()}")
            if torch.isnan(latent_tensor).any():
                print(f"NAN: {entry['audio_path']} has NaN values")
        except Exception as e:
            print(f"ERROR checking latent for {entry.get('audio_path', '<unknown>')}: {e}")

    # --- Report ---
    print("\n--- Validation Report ---")
    print(f"Total Entries Checked: {len(data):,}")
    print(f"✅ Valid Entries:       {len(valid_specs):,}")
    print(f"❌ Invalid Entries:     {len(invalid_logs):,}")
    print("-------------------------\n")

    # Write invalid text log
    if invalid_logs:
        with ERROR_LOG.open("w") as f:
            f.write("--- Vocal Dataset Validation Errors ---\n\n")
            f.write("\n".join(invalid_logs))
        print(f"Detailed invalid report saved to: {ERROR_LOG.resolve()}")
    else:
        print("🎉 All entries are valid under the current tolerances!")

    # Write valid specs CSV
    if valid_specs:
        fieldnames = [
            "audio_path",
            "group", "sub_group",
            "latent_path", "piano_roll_path", "encodec_path",
            "conditioning_amp_path", "conditioning_present",
            "vocal_lyrics_data_path", "vocal_lyrics_tensors_path", "vocal_syllable_boundaries_path",
            "vocal_conditioning_present",
            "latent_len", "conditioning_len", "piano_roll_len",
            "syllable_boundaries_len", "syllable_ok", "syllable_delta",
            "lyrics_syllable_count", "lyrics_tensors_keys",
            "lc_ok", "lc_delta", "lc_tol_abs", "lc_tol_rel_pct",
            "pr_ok", "pr_ref_len", "pr_shortfall", "pr_short_allow_abs", "pr_short_allow_rel_pct",
            "encodec_len", "encodec_expected", "encodec_delta", "encodec_tol_abs", "enc_ok",
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

if __name__ == "__main__":
    main()
