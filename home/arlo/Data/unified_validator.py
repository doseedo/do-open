#!/usr/bin/env python3
"""
Unified Validator - Combines all validation scripts into one master tool.

Combines functionality from:
- validate.py (random path validation)
- validate2.py (full validation with tolerances)
- validate3.py (rescue + single-bad nulling)
- finalvalidate.py (basic alignment checks)
- validate_conditioning_paths.py (conditioning path checks)
- validate_encodec_paths.py (encodec path rebuild)
- validate_vocal_manifest.py (vocal-specific validation)
- missing.py (find missing files by type)
- find_missing_conditioning.py (search for missing conditioning)
- filter_manifest_duration.py (duration filtering)

Usage:
    # Quick validation (25% random sample)
    python unified_validator.py --input manifest.json --mode quick

    # Full validation (all entries)
    python unified_validator.py --input manifest.json --mode full

    # Full validation with repair (fix what can be fixed)
    python unified_validator.py --input manifest.json --mode repair --output fixed_manifest.json

    # Validate for instrumental training
    python unified_validator.py --input manifest.json --mode full --type instrumental

    # Validate for vocal training
    python unified_validator.py --input manifest.json --mode full --type vocal

    # Filter by duration
    python unified_validator.py --input manifest.json --mode filter --min-duration 6.0 --output filtered.json
"""

import argparse
import csv
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Grid / hop params
DCAE_SR = 44100
DCAE_HOP_LENGTH = 4096

ENCODEC_SR = 24000
ENCODEC_HOP_LENGTH = 320

# Tolerances for slow grid (latent / conditioning / piano roll)
SLOW_SPREAD_ABS_FRAMES = 12
SLOW_SPREAD_REL_FRACTION = 0.06  # ~6%

# Piano roll can be shorter than reference
PR_SHORT_ABS_FRAMES = 24
PR_SHORT_REL_FRACTION = 0.18

# Encodec fast-grid tolerance
ENC_ABS_FLOOR_FRAMES = 24
ENC_REL_FRACTION = 0.015  # 1.5%

# Rescue rules for PR
HUGE_SHORTFALL_ABS_FRAMES = 96
HUGE_SHORTFALL_REL_FRACTION = 0.30

# Silence detection
COND_SILENCE_EPS = 1e-3
COND_SILENCE_FRACTION_QUIET = 0.98

# Conditioning types
COND_TYPES = ["amp", "rbend", "rframe", "onsets", "f0", "f0_masked"]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _within(a: int, b: int, abs_frames: int, rel_frac: float) -> bool:
    """Check if two values are within tolerance."""
    diff = abs(a - b)
    return (diff <= abs_frames) or (diff <= rel_frac * max(a, b, 1))


def _piano_roll_short_ok(pr_len: int, ref_len: int) -> bool:
    """Check if piano roll shortfall is acceptable."""
    if pr_len > ref_len:
        return False
    shortfall = ref_len - pr_len
    return shortfall <= max(PR_SHORT_ABS_FRAMES, int(PR_SHORT_REL_FRACTION * ref_len))


def _huge_shortfall(pr_len: int, ref_len: int) -> bool:
    """Check if shortfall is huge (requires silence proof)."""
    shortfall = ref_len - pr_len
    return shortfall > max(HUGE_SHORTFALL_ABS_FRAMES, int(HUGE_SHORTFALL_REL_FRACTION * ref_len))


def _safe_np_load(path: Path) -> np.ndarray:
    """Safely load numpy file."""
    if not path or str(path).strip() in {"", "."}:
        raise FileNotFoundError("empty path")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        return np.load(path)
    except ValueError:
        return np.load(path, allow_pickle=True)


def _first_tensor(x):
    """Recursively find first tensor in nested structure."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (list, tuple)):
        for item in x:
            t = _first_tensor(item)
            if t is not None:
                return t
    if isinstance(x, dict):
        for k in ("codes", "tokens", "encodec", "audio_tokens", "data", "latents"):
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
    """Extract latent length from blob."""
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
    """Extract encodec length from blob."""
    try:
        t = _first_tensor(obj)
        if t is None:
            return None
        return int(t.shape[-1])
    except Exception:
        return None


def _expected_encodec_len(latent_len: int) -> float:
    """Calculate expected encodec length from latent length."""
    slow_rate = DCAE_SR / DCAE_HOP_LENGTH
    fast_rate = ENCODEC_SR / ENCODEC_HOP_LENGTH
    return latent_len * (fast_rate / slow_rate)


def _ok_encodec_len(actual: int, expected: float) -> Tuple[bool, int]:
    """Check if encodec length is acceptable."""
    tol = max(ENC_ABS_FLOOR_FRAMES, int(ENC_REL_FRACTION * expected))
    return (abs(actual - expected) <= tol, tol)


def _cond_tail_is_silent(amp: np.ndarray, start: int, end: int) -> bool:
    """Check if conditioning tail is mostly silent."""
    if amp is None or amp.ndim == 0:
        return False
    start = max(start, 0)
    end = min(end, amp.shape[-1])
    if end <= start:
        return False
    tail = amp[..., start:end]
    if tail.size == 0:
        return False
    quiet = (tail <= COND_SILENCE_EPS)
    frac_quiet = float(np.count_nonzero(quiet)) / float(tail.size)
    return frac_quiet >= COND_SILENCE_FRACTION_QUIET


def _pr_tail_all_zero(pr: np.ndarray, start: int) -> bool:
    """Check if piano roll tail is all zeros."""
    if pr.ndim < 2:
        return False
    if start >= pr.shape[1]:
        return True
    tail = pr[:, start:]
    return np.all(tail == 0)


def has_dup_root(p: str) -> bool:
    """Check for duplicate /mnt/msdd/ in path."""
    return isinstance(p, str) and p.count("/mnt/msdd/") > 1


def exists_file(p: str) -> bool:
    """Check if file exists."""
    if not p or not isinstance(p, str):
        return False
    return Path(p).is_file()


# =============================================================================
# VALIDATION CLASSES
# =============================================================================

class ValidationResult:
    """Container for validation results."""

    def __init__(self, valid: bool, reason: str, specs: Dict[str, Any],
                 output_entry: Optional[Dict] = None, actions: List[str] = None):
        self.valid = valid
        self.reason = reason
        self.specs = specs
        self.output_entry = output_entry or {}
        self.actions = actions or []


class UnifiedValidator:
    """
    Unified validator for training manifests.
    Supports both instrumental and vocal training formats.
    """

    def __init__(self,
                 training_type: str = "instrumental",
                 conditioning_optional: bool = True,
                 repair_mode: bool = False,
                 pr_crop_dir: Optional[Path] = None):
        """
        Initialize validator.

        Args:
            training_type: "instrumental" or "vocal"
            conditioning_optional: If True, missing conditioning won't invalidate
            repair_mode: If True, attempt to fix issues (crop PR, null bad paths)
            pr_crop_dir: Directory to save cropped piano rolls
        """
        self.training_type = training_type
        self.conditioning_optional = conditioning_optional
        self.repair_mode = repair_mode
        self.pr_crop_dir = pr_crop_dir

        if pr_crop_dir:
            pr_crop_dir.mkdir(parents=True, exist_ok=True)

    def validate_entry(self, entry: Dict) -> ValidationResult:
        """
        Validate a single manifest entry.

        Returns ValidationResult with valid flag, reason, specs, and optionally repaired entry.
        """
        specs: Dict[str, Any] = {}
        out = dict(entry)
        actions: List[str] = []

        try:
            audio_path = entry.get("audio_path", "<unknown>")
            latent_path_str = entry.get("latent_path", "")
            encodec_path_str = entry.get("encodec_path", "")
            pr_path_str = entry.get("piano_roll_path", "")

            latent_path = Path(latent_path_str) if latent_path_str else None
            encodec_path = Path(encodec_path_str) if encodec_path_str else None
            pr_path = Path(pr_path_str) if pr_path_str else None

            cond_paths = entry.get("conditioning_paths") or {}
            amp_str = cond_paths.get("amp") if isinstance(cond_paths, dict) else None

            specs.update({
                "audio_path": audio_path,
                "latent_path": str(latent_path) if latent_path else "",
                "encodec_path": str(encodec_path) if encodec_path else "",
                "piano_roll_path": str(pr_path) if pr_path else "",
                "conditioning_amp_path": amp_str or "",
            })

            # Check for duplicate root paths
            for field in ["latent_path", "encodec_path", "piano_roll_path"]:
                val = entry.get(field, "")
                if has_dup_root(val):
                    return ValidationResult(False, f"Duplicate /mnt/msdd/ in {field}", specs)

            # Required path presence
            if not (latent_path and pr_path and encodec_path):
                return ValidationResult(False, "Missing latent, encodec, or piano_roll path", specs)

            # File existence
            if not latent_path.exists():
                return ValidationResult(False, "DCAE latent file missing", specs)

            # Load latent length (anchor)
            try:
                latent_blob = torch.load(latent_path, map_location="cpu")
                latent_len = _extract_latent_len(latent_blob)
                if latent_len is None:
                    return ValidationResult(False, "Could not read latent length", specs)
                specs["latent_len"] = int(latent_len)
            except Exception as e:
                return ValidationResult(False, f"Error loading latent: {e}", specs)

            # Check for latent quality issues
            try:
                latent_tensor = latent_blob.get("latents", latent_blob) if isinstance(latent_blob, dict) else latent_blob
                if isinstance(latent_tensor, torch.Tensor):
                    if torch.isnan(latent_tensor).any():
                        specs["has_nan"] = True
                        return ValidationResult(False, "Latent contains NaN values", specs)
                    if latent_tensor.abs().max() > 20:
                        specs["max_latent_value"] = float(latent_tensor.abs().max())
                        # Warning only, don't fail
                        actions.append(f"warning_latent_outlier_{float(latent_tensor.abs().max()):.1f}")
            except Exception:
                pass

            # Load conditioning if present
            cond_present = False
            cond_usable = False
            amp_np = None
            cond_len = None

            if isinstance(amp_str, str) and amp_str.strip():
                amp_path = Path(amp_str)
                if amp_path.exists() and amp_path.is_file():
                    try:
                        amp_np = _safe_np_load(amp_path)
                        cond_len = int(amp_np.shape[-1]) if amp_np.ndim >= 1 else int(len(amp_np))
                        cond_present = True
                        specs["conditioning_len"] = int(cond_len)

                        # Check alignment
                        lc_ok = _within(latent_len, cond_len, SLOW_SPREAD_ABS_FRAMES, SLOW_SPREAD_REL_FRACTION)
                        specs["lc_ok"] = bool(lc_ok)
                        specs["lc_delta"] = int(abs(latent_len - cond_len))
                        cond_usable = lc_ok
                    except Exception as e:
                        specs["conditioning_error"] = str(e)

            specs["conditioning_present"] = cond_present
            specs["conditioning_usable"] = cond_usable

            # Reference length for other checks
            ref_len = max(latent_len, cond_len) if cond_usable else latent_len
            specs["ref_len"] = int(ref_len)

            # Piano roll validation
            pr_ok = False
            pr_action = None

            if not pr_path.exists():
                if self.repair_mode:
                    out["piano_roll_path"] = None
                    actions.append("null_pr_path_missing")
                else:
                    return ValidationResult(False, "Piano roll file missing", specs)
            else:
                try:
                    pr = _safe_np_load(pr_path)
                    if pr.ndim < 2:
                        return ValidationResult(False, f"Piano roll has bad shape {pr.shape}", specs)
                    pr_len = int(pr.shape[1])
                    specs["piano_roll_len"] = pr_len

                    if pr_len < ref_len:
                        if _piano_roll_short_ok(pr_len, ref_len):
                            pr_ok = True
                        elif cond_usable and _cond_tail_is_silent(amp_np, pr_len, ref_len):
                            pr_ok = True
                            pr_action = "rescue_pr_short_tail_silent"
                        elif self.repair_mode:
                            out["piano_roll_path"] = None
                            actions.append("null_pr_path_short")
                        else:
                            specs["pr_shortfall"] = int(ref_len - pr_len)
                            return ValidationResult(False, f"Piano roll too short ({pr_len} vs {ref_len})", specs)
                    elif pr_len == ref_len:
                        pr_ok = True
                    else:
                        # PR longer - check for zero tail
                        if _pr_tail_all_zero(pr, ref_len):
                            if self.repair_mode and self.pr_crop_dir:
                                try:
                                    cropped = pr[:, :ref_len]
                                    crop_path = self.pr_crop_dir / (pr_path.stem + f".crop{ref_len}" + pr_path.suffix)
                                    np.save(crop_path, cropped)
                                    out["piano_roll_path"] = str(crop_path)
                                    pr_ok = True
                                    pr_action = f"crop_pr_tail_to_{ref_len}"
                                except Exception as e:
                                    pr_action = f"crop_failed:{e}"
                            else:
                                pr_ok = True  # Accept if tail is zeros
                        elif self.repair_mode:
                            out["piano_roll_path"] = None
                            actions.append("null_pr_path_long")
                        else:
                            return ValidationResult(False, f"Piano roll too long ({pr_len} vs {ref_len})", specs)

                    if pr_action:
                        actions.append(pr_action)

                except Exception as e:
                    if self.repair_mode:
                        out["piano_roll_path"] = None
                        actions.append(f"null_pr_load_error")
                    else:
                        return ValidationResult(False, f"Error loading piano roll: {e}", specs)

            specs["pr_ok"] = pr_ok or (self.repair_mode and out.get("piano_roll_path") is None)

            # Encodec validation
            enc_ok = False

            if not encodec_path.exists():
                if self.repair_mode:
                    out["encodec_path"] = None
                    actions.append("null_encodec_path_missing")
                else:
                    return ValidationResult(False, "Encodec token file missing", specs)
            else:
                try:
                    enc_blob = torch.load(encodec_path, map_location="cpu")
                    encodec_len = _extract_encodec_len(enc_blob)

                    if encodec_len is None:
                        if self.repair_mode:
                            out["encodec_path"] = None
                            actions.append("null_encodec_unreadable")
                        else:
                            return ValidationResult(False, "Could not read encodec length", specs)
                    else:
                        specs["encodec_len"] = int(encodec_len)
                        expected_enc = _expected_encodec_len(latent_len)
                        enc_ok, tol = _ok_encodec_len(encodec_len, expected_enc)
                        specs["encodec_expected"] = int(round(expected_enc))
                        specs["encodec_delta"] = int(abs(encodec_len - int(round(expected_enc))))
                        specs["encodec_tol"] = int(tol)

                        if not enc_ok:
                            if self.repair_mode:
                                out["encodec_path"] = None
                                actions.append("null_encodec_misaligned")
                            else:
                                return ValidationResult(False,
                                    f"Encodec length mismatch ({encodec_len} vs expected {int(expected_enc)})", specs)
                except Exception as e:
                    if self.repair_mode:
                        out["encodec_path"] = None
                        actions.append("null_encodec_load_error")
                    else:
                        return ValidationResult(False, f"Error loading encodec: {e}", specs)

            specs["enc_ok"] = enc_ok or (self.repair_mode and out.get("encodec_path") is None)

            # Conditioning - drop if misaligned (optional)
            if cond_present and not cond_usable:
                if self.conditioning_optional or self.repair_mode:
                    out["conditioning_paths"] = {}
                    actions.append("drop_conditioning_mismatch")
                else:
                    return ValidationResult(False,
                        f"Conditioning alignment error ({cond_len} vs latent {latent_len})", specs)

            # Vocal-specific validation
            if self.training_type == "vocal":
                vocal_result = self._validate_vocal_conditioning(entry, specs, latent_len)
                if not vocal_result.valid:
                    if self.repair_mode:
                        out["vocal_conditioning_paths"] = {}
                        actions.append("drop_vocal_conditioning")
                    else:
                        return vocal_result

            # Final acceptance check in repair mode
            if self.repair_mode:
                # Count nulled paths
                nulled = sum([
                    out.get("piano_roll_path") is None,
                    out.get("encodec_path") is None,
                ])
                if nulled > 1:
                    return ValidationResult(False, "Too many paths needed repair (>1)", specs)

                actions.append("accept_repaired" if actions else "accept_normal")
            else:
                actions.append("accept_normal")

            specs["actions"] = actions
            return ValidationResult(True, "Valid", specs, out, actions)

        except Exception as e:
            return ValidationResult(False, f"Error processing entry: {e}", specs)

    def _validate_vocal_conditioning(self, entry: Dict, specs: Dict, latent_len: int) -> ValidationResult:
        """Validate vocal-specific conditioning."""
        vocal_cond = entry.get("vocal_conditioning_paths") or {}

        if not isinstance(vocal_cond, dict):
            return ValidationResult(False, "vocal_conditioning_paths not a dict", specs)

        lyrics_data_str = vocal_cond.get("lyrics_data")
        lyrics_tensors_str = vocal_cond.get("lyrics_tensors")
        syllable_boundaries_str = vocal_cond.get("syllable_boundaries")

        # Check presence
        if not all([lyrics_data_str, lyrics_tensors_str, syllable_boundaries_str]):
            return ValidationResult(False, "Vocal conditioning paths incomplete", specs)

        # Check existence
        for name, path_str in [("lyrics_data", lyrics_data_str),
                               ("lyrics_tensors", lyrics_tensors_str),
                               ("syllable_boundaries", syllable_boundaries_str)]:
            if not Path(path_str).exists():
                return ValidationResult(False, f"Vocal conditioning {name} missing", specs)

        # Check syllable boundaries alignment
        try:
            syllable_data = _safe_np_load(Path(syllable_boundaries_str))
            syllable_len = int(syllable_data.shape[-1]) if syllable_data.ndim >= 1 else int(len(syllable_data))
            specs["syllable_len"] = syllable_len

            syll_ok = _within(latent_len, syllable_len, SLOW_SPREAD_ABS_FRAMES, SLOW_SPREAD_REL_FRACTION)
            specs["syllable_ok"] = syll_ok

            if not syll_ok:
                return ValidationResult(False,
                    f"Syllable boundaries mismatch ({syllable_len} vs latent {latent_len})", specs)
        except Exception as e:
            return ValidationResult(False, f"Error loading syllable boundaries: {e}", specs)

        # Validate lyrics_data JSON
        try:
            with open(lyrics_data_str, 'r') as f:
                lyrics_json = json.load(f)
            if "syllable_timings" not in lyrics_json:
                return ValidationResult(False, "lyrics_data missing syllable_timings", specs)
            specs["lyrics_syllable_count"] = len(lyrics_json.get("syllable_timings", []))
        except Exception as e:
            return ValidationResult(False, f"Error loading lyrics_data: {e}", specs)

        # Validate lyrics_tensors
        try:
            lyrics_tensors = torch.load(lyrics_tensors_str, map_location="cpu")
            if not isinstance(lyrics_tensors, dict):
                return ValidationResult(False, "lyrics_tensors should be a dict", specs)
            expected_keys = {"lyrics_indices", "lyrics_embeddings", "phoneme_embeddings"}
            missing = expected_keys - set(lyrics_tensors.keys())
            if missing:
                return ValidationResult(False, f"lyrics_tensors missing keys: {missing}", specs)
        except Exception as e:
            return ValidationResult(False, f"Error loading lyrics_tensors: {e}", specs)

        specs["vocal_conditioning_valid"] = True
        return ValidationResult(True, "Vocal conditioning valid", specs)

    def validate_manifest(self, manifest: List[Dict],
                         sample_fraction: float = 1.0,
                         seed: int = 42) -> Dict[str, Any]:
        """
        Validate entire manifest.

        Args:
            manifest: List of manifest entries
            sample_fraction: Fraction of entries to validate (1.0 = all)
            seed: Random seed for sampling

        Returns:
            Dict with results: valid_entries, invalid_entries, stats, etc.
        """
        # Sample if needed
        if sample_fraction < 1.0:
            n_sample = max(1, int(len(manifest) * sample_fraction))
            rng = random.Random(seed)
            indices = rng.sample(range(len(manifest)), n_sample)
            entries_to_check = [(i, manifest[i]) for i in indices]
        else:
            entries_to_check = list(enumerate(manifest))

        valid_entries = []
        invalid_entries = []
        repaired_entries = []
        all_specs = []
        actions_log = []

        stats = {
            "total": len(manifest),
            "checked": len(entries_to_check),
            "valid": 0,
            "invalid": 0,
            "repaired": 0,
            "issues": defaultdict(int),
        }

        for idx, entry in tqdm(entries_to_check, desc="Validating"):
            result = self.validate_entry(entry)

            if result.valid:
                stats["valid"] += 1
                valid_entries.append(result.output_entry)

                if result.actions and any(a not in ("accept_normal",) for a in result.actions):
                    stats["repaired"] += 1
                    repaired_entries.append({
                        "index": idx,
                        "audio_path": entry.get("audio_path"),
                        "actions": result.actions
                    })
                    actions_log.append(f"{entry.get('audio_path', f'idx:{idx}')}: {', '.join(result.actions)}")
            else:
                stats["invalid"] += 1
                invalid_entries.append({
                    "index": idx,
                    "audio_path": entry.get("audio_path"),
                    "reason": result.reason,
                    "specs": result.specs
                })

                # Track issue types
                reason_key = result.reason.split(":")[0].strip() if ":" in result.reason else result.reason
                stats["issues"][reason_key] += 1

            all_specs.append(result.specs)

        return {
            "stats": dict(stats),
            "valid_entries": valid_entries,
            "invalid_entries": invalid_entries,
            "repaired_entries": repaired_entries,
            "actions_log": actions_log,
            "all_specs": all_specs,
        }


# =============================================================================
# ADDITIONAL UTILITIES
# =============================================================================

def filter_by_duration(manifest: List[Dict], min_duration: float = 6.0) -> Tuple[List[Dict], List[Dict]]:
    """
    Filter manifest by audio duration.

    Returns: (kept_entries, removed_entries)
    """
    kept = []
    removed = []

    for entry in tqdm(manifest, desc="Checking durations"):
        audio_path = entry.get("audio_path")

        if not audio_path or not Path(audio_path).exists():
            kept.append(entry)  # Keep entries without audio
            continue

        try:
            info = torchaudio.info(audio_path)
            duration = info.num_frames / info.sample_rate

            if duration >= min_duration:
                kept.append(entry)
            else:
                removed.append({
                    "entry": entry,
                    "duration": duration
                })
        except Exception as e:
            kept.append(entry)  # Keep if can't read

    return kept, removed


def find_missing_by_type(manifest: List[Dict], feature_types: List[str] = None) -> Dict[str, List[str]]:
    """
    Find entries missing specific feature types.

    Args:
        manifest: List of manifest entries
        feature_types: List of feature types to check:
            ["latent", "encodec", "piano_roll", "conditioning", "audio"]

    Returns:
        Dict mapping feature type to list of audio_paths with missing features
    """
    if feature_types is None:
        feature_types = ["latent", "encodec", "piano_roll", "conditioning", "audio"]

    missing = {ft: [] for ft in feature_types}

    for entry in tqdm(manifest, desc="Finding missing"):
        audio_path = entry.get("audio_path", "<unknown>")

        if "audio" in feature_types:
            ap = entry.get("audio_path")
            if not ap or not Path(ap).exists():
                missing["audio"].append(audio_path)

        if "latent" in feature_types:
            lp = entry.get("latent_path")
            if not lp or not Path(lp).exists():
                missing["latent"].append(audio_path)

        if "encodec" in feature_types:
            ep = entry.get("encodec_path")
            if not ep or not Path(ep).exists():
                missing["encodec"].append(audio_path)

        if "piano_roll" in feature_types:
            pp = entry.get("piano_roll_path")
            if not pp or not Path(pp).exists():
                missing["piano_roll"].append(audio_path)

        if "conditioning" in feature_types:
            cond = entry.get("conditioning_paths") or {}
            missing_cond = False
            for ct in COND_TYPES:
                cp = cond.get(ct)
                if not cp or not Path(cp).exists():
                    missing_cond = True
                    break
            if missing_cond:
                missing["conditioning"].append(audio_path)

    return missing


def write_validation_report(results: Dict[str, Any], output_dir: Path, prefix: str = "validation"):
    """Write validation results to files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stats = results["stats"]

    # Summary
    summary_path = output_dir / f"{prefix}_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Total entries:    {stats['total']:,}\n")
        f.write(f"Entries checked:  {stats['checked']:,}\n")
        f.write(f"Valid entries:    {stats['valid']:,}\n")
        f.write(f"Invalid entries:  {stats['invalid']:,}\n")
        f.write(f"Repaired entries: {stats['repaired']:,}\n")
        f.write(f"\nValidation rate:  {100*stats['valid']/max(1,stats['checked']):.1f}%\n")

        if stats["issues"]:
            f.write("\nIssue breakdown:\n")
            for issue, count in sorted(stats["issues"].items(), key=lambda x: -x[1]):
                f.write(f"  {issue}: {count}\n")

    print(f"Summary saved to: {summary_path}")

    # Invalid entries log
    if results["invalid_entries"]:
        errors_path = output_dir / f"{prefix}_errors.txt"
        with open(errors_path, "w") as f:
            f.write("--- Validation Errors ---\n\n")
            for item in results["invalid_entries"]:
                f.write(f"File: {item['audio_path']}\n")
                f.write(f"Reason: {item['reason']}\n\n")
        print(f"Errors saved to: {errors_path}")

    # Actions log
    if results["actions_log"]:
        actions_path = output_dir / f"{prefix}_actions.txt"
        with open(actions_path, "w") as f:
            f.write("--- Repair Actions ---\n\n")
            f.write("\n".join(results["actions_log"]))
        print(f"Actions saved to: {actions_path}")

    # CSV of all specs
    if results["all_specs"]:
        csv_path = output_dir / f"{prefix}_specs.csv"

        # Get all possible keys
        all_keys = set()
        for spec in results["all_specs"]:
            all_keys.update(spec.keys())
        fieldnames = sorted(all_keys)

        try:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for spec in results["all_specs"]:
                    row = {k: spec.get(k, "") for k in fieldnames}
                    writer.writerow(row)
            print(f"Specs CSV saved to: {csv_path}")
        except Exception as e:
            print(f"Warning: Could not write CSV: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified Validator for training manifests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick validation (25% sample)
  python unified_validator.py --input manifest.json --mode quick

  # Full validation
  python unified_validator.py --input manifest.json --mode full

  # Full validation with repair
  python unified_validator.py --input manifest.json --mode repair --output fixed.json

  # Vocal manifest validation
  python unified_validator.py --input vocal_manifest.json --type vocal

  # Filter by duration
  python unified_validator.py --input manifest.json --mode filter --min-duration 6.0
        """
    )

    parser.add_argument("--input", "-i", required=True, help="Input manifest JSON")
    parser.add_argument("--output", "-o", help="Output manifest JSON (for repair/filter modes)")
    parser.add_argument("--mode", choices=["quick", "full", "repair", "filter", "find-missing"],
                       default="full", help="Validation mode")
    parser.add_argument("--type", choices=["instrumental", "vocal"], default="instrumental",
                       help="Training type")
    parser.add_argument("--sample", type=float, default=0.25,
                       help="Sample fraction for quick mode (default: 0.25)")
    parser.add_argument("--min-duration", type=float, default=6.0,
                       help="Minimum duration in seconds (for filter mode)")
    parser.add_argument("--output-dir", type=Path, default=Path("./validation_logs"),
                       help="Directory for validation reports")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--conditioning-optional", action="store_true",
                       help="Treat missing conditioning as OK")

    args = parser.parse_args()

    # Load manifest
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    print(f"Loading manifest: {input_path}")
    with open(input_path, "r") as f:
        manifest = json.load(f)
    print(f"Loaded {len(manifest):,} entries")

    # Handle different modes
    if args.mode == "filter":
        print(f"\nFiltering by duration >= {args.min_duration}s...")
        kept, removed = filter_by_duration(manifest, args.min_duration)

        print(f"\nResults:")
        print(f"  Kept:    {len(kept):,}")
        print(f"  Removed: {len(removed):,}")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(kept, f, indent=2)
            print(f"\nSaved to: {output_path}")

        return 0

    elif args.mode == "find-missing":
        print("\nFinding missing features...")
        missing = find_missing_by_type(manifest)

        print("\nResults:")
        for ft, paths in missing.items():
            print(f"  {ft}: {len(paths):,} missing")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        for ft, paths in missing.items():
            if paths:
                out_path = args.output_dir / f"missing_{ft}.txt"
                with open(out_path, "w") as f:
                    f.write("\n".join(paths))
                print(f"  Saved: {out_path}")

        return 0

    else:
        # Validation modes
        repair_mode = args.mode == "repair"
        sample_fraction = args.sample if args.mode == "quick" else 1.0

        pr_crop_dir = args.output_dir / "rescued_pianorolls" if repair_mode else None

        validator = UnifiedValidator(
            training_type=args.type,
            conditioning_optional=args.conditioning_optional or args.type == "instrumental",
            repair_mode=repair_mode,
            pr_crop_dir=pr_crop_dir
        )

        print(f"\nValidating ({args.mode} mode, {args.type} type)...")
        results = validator.validate_manifest(manifest, sample_fraction, args.seed)

        # Print summary
        stats = results["stats"]
        print("\n" + "=" * 70)
        print("VALIDATION RESULTS")
        print("=" * 70)
        print(f"Total entries:    {stats['total']:,}")
        print(f"Entries checked:  {stats['checked']:,}")
        print(f"Valid entries:    {stats['valid']:,} ({100*stats['valid']/max(1,stats['checked']):.1f}%)")
        print(f"Invalid entries:  {stats['invalid']:,}")
        if repair_mode:
            print(f"Repaired entries: {stats['repaired']:,}")

        if stats["issues"]:
            print("\nTop issues:")
            for issue, count in sorted(stats["issues"].items(), key=lambda x: -x[1])[:10]:
                print(f"  {issue}: {count}")

        # Write reports
        write_validation_report(results, args.output_dir)

        # Save repaired manifest
        if repair_mode and args.output and results["valid_entries"]:
            output_path = Path(args.output)
            with open(output_path, "w") as f:
                json.dump(results["valid_entries"], f, indent=2)
            print(f"\nRepaired manifest saved to: {output_path}")
            print(f"  Entries: {len(results['valid_entries']):,}")

        return 0


if __name__ == "__main__":
    exit(main())
