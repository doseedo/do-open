#!/usr/bin/env python3
"""
Manifest Quality Estimator for PerformerCN2 Training Data

Analyzes a training manifest and produces detailed quality metrics to
estimate how well a model will train on the data. Checks:
  1. File availability (all streams present)
  2. Shape validation (latents, encodec, piano_roll, conditioning)
  3. Label quality (group/subgroup correctness, distribution)
  4. Signal quality (empty signals, NaN, value ranges, pitch consistency)
  5. Cross-signal consistency (PR↔f0, PR↔amp, onset alignment)
  6. Per-group and overall quality scores

Usage:
  python3 manifest_quality_estimator.py <manifest.json> [--sample N] [--workers N]

The manifest can use old /mnt/msdd paths or new bucket paths.
Old paths are auto-remapped to /home/arlo/gcs-bucket/.
"""

import argparse
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import orjson
import torch

# ── Vocab (from dataloader.py) ──────────────────────────────────────────────
APPROVED_GROUPS = ["piano", "guitar", "bass", "strings", "brass", "winds"]
APPROVED_SUBGROUPS = {
    "piano":   ["acoustic_piano", "keys", "undefined"],
    "guitar":  ["acoustic_guitar", "electric_guitar", "plucked", "undefined"],
    "bass":    ["electric_bass", "upright_bass", "undefined"],
    "strings": ["violin", "viola", "cello", "undefined"],
    "brass":   ["trumpet", "trombone", "french_horn", "tuba", "undefined"],
    "winds":   ["bassoon", "clarinet", "flute", "oboe", "sax"],
}
ALL_SUBGROUPS = sorted({sg for subs in APPROVED_SUBGROUPS.values() for sg in subs})

# Expected pitch ranges (MIDI note numbers) per group
# Generous ranges to catch only egregious mislabels
PITCH_RANGES = {
    "piano":   (21, 108),   # A0 to C8
    "guitar":  (28, 96),    # E1 to C7 (harmonics)
    "bass":    (20, 72),    # E0 to C5
    "strings": (36, 105),   # C2 to A7
    "brass":   (34, 96),    # Bb1 to C7
    "winds":   (36, 108),   # C2 to C8
}

# Monophonic expectation: these groups are mostly monophonic
MONO_GROUPS = {"brass", "winds"}

DCAE_SR, DCAE_HOP = 44100, 4096
SLOW_HZ = DCAE_SR / DCAE_HOP  # ~10.77
ENC_SR, ENC_HOP = 24000, 320
FAST_PER_SLOW = (ENC_SR / ENC_HOP) / SLOW_HZ

BUCKET = "/home/arlo/gcs-bucket"


# ── Path remapping ───────────────────────────────────────────────────────────

def remap_latent_path(p: str) -> str:
    """  /mnt/msdd/dcae_latentsnew/date/New/session/Audio Files/file.pt
      -> /home/arlo/gcs-bucket/Latents/protools/date/New/session/Audio Files/file.pt
    """
    if not p or not p.startswith("/mnt/msdd/dcae_latentsnew/"):
        return p
    rest = p[len("/mnt/msdd/dcae_latentsnew/"):]
    return f"{BUCKET}/Latents/protools/{rest}"


def remap_encodec_path(p: str) -> str:
    """  /mnt/msdd/encodec_tokens/session/file.pt
      -> /home/arlo/gcs-bucket/Tokens/session/file.pt
    """
    if not p or not p.startswith("/mnt/msdd/encodec_tokens/"):
        return p
    rest = p[len("/mnt/msdd/encodec_tokens/"):]
    return f"{BUCKET}/Tokens/{rest}"


def remap_conditioning_path(p: str) -> str:
    """  /mnt/msdd/{more,evenmore,new}conditioning/date/New/session/Audio Files/file.ext.npy
      -> /home/arlo/gcs-bucket/Conditioning/protools/date/New/session/Audio Files/file.ext.npy
    """
    if not p:
        return p
    for prefix in ["/mnt/msdd/moreconditioning/",
                   "/mnt/msdd/evenmoreconditioning/",
                   "/mnt/msdd/newconditioning/"]:
        if p.startswith(prefix):
            rest = p[len(prefix):]
            return f"{BUCKET}/Conditioning/protools/{rest}"
    return p


def remap_piano_roll_path(pr_path: str, latent_path: str) -> str:
    """  /mnt/msdd/piano_rolls/session/file.pianoroll.npy
      -> derive from latent_path: /home/arlo/gcs-bucket/Conditioning/protools/date/New/session/Audio Files/file.pianoroll.npy

    The old piano_roll path is flat (session/file) while in the bucket
    they're in the Conditioning folder with the date structure from latent_path.
    """
    if not pr_path or not pr_path.startswith("/mnt/msdd/piano_rolls/"):
        return pr_path
    # Extract filename from pr_path
    pr_filename = os.path.basename(pr_path)
    # Get the date/New/session/Audio Files/ structure from the latent path
    remapped_lat = remap_latent_path(latent_path)
    if remapped_lat and "/Latents/protools/" in remapped_lat:
        lat_dir = os.path.dirname(remapped_lat)
        # lat_dir = .../Latents/protools/date/New/session/Audio Files
        # Replace Latents with Conditioning
        cond_dir = lat_dir.replace("/Latents/", "/Conditioning/")
        return os.path.join(cond_dir, pr_filename)
    return pr_path


def remap_entry(entry: dict) -> dict:
    """Remap all paths in a manifest entry from /mnt/msdd to bucket paths."""
    e = dict(entry)
    lat = e.get("latent_path", "")
    e["latent_path"] = remap_latent_path(lat) if lat else lat
    e["encodec_path"] = remap_encodec_path(e.get("encodec_path", "")) if e.get("encodec_path") else e.get("encodec_path")
    e["piano_roll_path"] = remap_piano_roll_path(e.get("piano_roll_path", ""), lat) if e.get("piano_roll_path") else e.get("piano_roll_path")
    cp = e.get("conditioning_paths")
    if cp:
        e["conditioning_paths"] = {k: remap_conditioning_path(v) for k, v in cp.items()}
    return e


# ── Safe loaders ─────────────────────────────────────────────────────────────

def safe_np(path: str) -> Optional[np.ndarray]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        return np.load(p)
    except Exception:
        return None


def safe_pt(path: str) -> Optional[torch.Tensor]:
    if not path:
        return None
    try:
        p = Path(path)
        if not p.exists():
            return None
        obj = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.Tensor):
            return obj
        if isinstance(obj, dict):
            for k in ("latents", "codes", "tokens", "encodec", "audio_tokens", "data"):
                if k in obj and isinstance(obj[k], torch.Tensor):
                    return obj[k]
        # recursively find first tensor
        def first_t(x):
            if isinstance(x, torch.Tensor):
                return x
            if isinstance(x, (list, tuple)):
                for it in x:
                    t = first_t(it)
                    if t is not None:
                        return t
            if isinstance(x, dict):
                for v in x.values():
                    t = first_t(v)
                    if t is not None:
                        return t
            return None
        return first_t(obj)
    except Exception:
        return None


# ── Per-entry analysis ───────────────────────────────────────────────────────

def analyze_entry(entry: dict, idx: int) -> dict:
    """Analyze a single manifest entry. Returns a dict of metrics."""
    result = {
        "idx": idx,
        "group": (entry.get("group") or "MISSING").lower(),
        "sub_group": (entry.get("sub_group") or "MISSING").lower(),
        "issues": [],
    }

    group = result["group"]
    subgroup = result["sub_group"]

    # ── Label checks ──
    if group not in APPROVED_GROUPS:
        result["issues"].append(("label", f"invalid group: {group}"))
        result["label_group_valid"] = False
    else:
        result["label_group_valid"] = True

    valid_subs = APPROVED_SUBGROUPS.get(group, [])
    if subgroup not in valid_subs and subgroup not in ALL_SUBGROUPS:
        result["issues"].append(("label", f"invalid subgroup: {subgroup} for group {group}"))
        result["label_sub_valid"] = False
    elif subgroup not in valid_subs:
        result["issues"].append(("label", f"subgroup '{subgroup}' mismatched to group '{group}'"))
        result["label_sub_valid"] = False
    else:
        result["label_sub_valid"] = True

    if subgroup == "undefined":
        result["label_sub_undefined"] = True
    else:
        result["label_sub_undefined"] = False

    # ── File availability ──
    for ftype in ["latent_path", "encodec_path", "piano_roll_path"]:
        p = entry.get(ftype)
        result[f"{ftype}_present"] = p is not None and p != ""
        result[f"{ftype}_exists"] = os.path.exists(p) if p else False

    cond_paths = entry.get("conditioning_paths") or {}
    for ckey in ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]:
        cp = cond_paths.get(ckey)
        result[f"cond_{ckey}_exists"] = os.path.exists(cp) if cp else False

    # ── Load and validate latents ──
    lat = safe_pt(entry.get("latent_path"))
    if lat is None:
        result["latent_loaded"] = False
        result["issues"].append(("file", "latent failed to load"))
        return result
    result["latent_loaded"] = True

    if lat.dim() != 3:
        result["issues"].append(("shape", f"latent dim={lat.dim()}, expected 3"))
        result["latent_shape_ok"] = False
        return result
    if lat.shape[0] != 8 or lat.shape[1] != 16:
        result["issues"].append(("shape", f"latent shape {list(lat.shape)}, expected [8,16,T]"))
        result["latent_shape_ok"] = False
    else:
        result["latent_shape_ok"] = True

    T_slow = lat.shape[2]
    result["T_slow"] = T_slow
    result["duration_sec"] = T_slow / SLOW_HZ

    # Latent stats
    lat_f = lat.float()
    result["latent_mean"] = lat_f.mean().item()
    result["latent_std"] = lat_f.std().item()
    result["latent_min"] = lat_f.min().item()
    result["latent_max"] = lat_f.max().item()
    result["latent_has_nan"] = bool(torch.isnan(lat_f).any())
    result["latent_has_inf"] = bool(torch.isinf(lat_f).any())
    result["latent_all_zero"] = bool((lat_f == 0).all())

    if result["latent_has_nan"]:
        result["issues"].append(("signal", "latent contains NaN"))
    if result["latent_has_inf"]:
        result["issues"].append(("signal", "latent contains Inf"))
    if result["latent_all_zero"]:
        result["issues"].append(("signal", "latent is all zeros (silent/corrupt)"))
    if result["latent_std"] < 0.001:
        result["issues"].append(("signal", f"latent std very low ({result['latent_std']:.6f})"))

    # ── Load and validate encodec ──
    enc = safe_pt(entry.get("encodec_path"))
    if enc is None:
        result["encodec_loaded"] = False
        result["issues"].append(("file", "encodec failed to load"))
    else:
        result["encodec_loaded"] = True
        if enc.dim() == 3 and enc.shape[0] == 1:
            enc = enc.squeeze(0)
        if enc.dim() != 2:
            result["issues"].append(("shape", f"encodec dim={enc.dim()}, expected 2"))
            result["encodec_shape_ok"] = False
        else:
            if enc.shape[0] > 8:
                enc = enc[:8]
            result["encodec_shape_ok"] = enc.shape[0] <= 8
            T_fast = enc.shape[1]
            result["T_fast"] = T_fast
            expected_ratio = T_fast / T_slow if T_slow > 0 else 0
            result["fast_slow_ratio"] = expected_ratio
            ratio_err = abs(expected_ratio - FAST_PER_SLOW) / FAST_PER_SLOW if FAST_PER_SLOW > 0 else 0
            if ratio_err > 0.2:
                result["issues"].append(("alignment", f"fast/slow ratio {expected_ratio:.2f}, expected ~{FAST_PER_SLOW:.2f}"))
            result["encodec_all_zero"] = bool((enc == 0).all())
            if result["encodec_all_zero"]:
                result["issues"].append(("signal", "encodec tokens all zero (no timbre info)"))

    # ── Load and validate piano roll ──
    pr_np = safe_np(entry.get("piano_roll_path"))
    if pr_np is None:
        result["piano_roll_loaded"] = False
        if entry.get("piano_roll_path"):
            result["issues"].append(("file", "piano_roll failed to load"))
    else:
        result["piano_roll_loaded"] = True
        pr = torch.from_numpy(pr_np).float()
        if pr.dim() != 2 or pr.shape[0] != 128:
            result["issues"].append(("shape", f"piano_roll shape {list(pr.shape)}, expected [128,T]"))
            result["pr_shape_ok"] = False
        else:
            result["pr_shape_ok"] = True
            pr_T = pr.shape[1]
            result["pr_T"] = pr_T
            # Temporal alignment with latent
            if abs(pr_T - T_slow) > max(5, T_slow * 0.05):
                result["issues"].append(("alignment", f"piano_roll T={pr_T} vs latent T={T_slow}"))

            # Activity analysis
            pr_active = (pr > 0)
            active_frames = pr_active.any(dim=0).float()
            result["pr_activity_ratio"] = active_frames.mean().item()
            result["pr_all_zero"] = bool(not pr_active.any())

            if result["pr_all_zero"]:
                result["issues"].append(("signal", "piano_roll is all zeros (no MIDI info)"))
            elif result["pr_activity_ratio"] < 0.02:
                result["issues"].append(("signal", f"piano_roll very sparse ({result['pr_activity_ratio']:.3f} active)"))

            # Note density
            notes_per_frame = pr_active.float().sum(dim=0)
            result["pr_mean_polyphony"] = notes_per_frame[notes_per_frame > 0].mean().item() if notes_per_frame.any() else 0

            # Polyphony check for mono instruments
            if group in MONO_GROUPS and result["pr_mean_polyphony"] > 2.5:
                result["issues"].append(("label", f"mono group '{group}' has mean polyphony {result['pr_mean_polyphony']:.1f}"))

            # Pitch range check
            if pr_active.any():
                active_pitches = pr_active.any(dim=1).nonzero(as_tuple=False).view(-1)
                p_lo, p_hi = active_pitches.min().item(), active_pitches.max().item()
                result["pr_pitch_lo"] = p_lo
                result["pr_pitch_hi"] = p_hi

                if group in PITCH_RANGES:
                    exp_lo, exp_hi = PITCH_RANGES[group]
                    # Check if most of the pitch content is way outside expected range
                    out_of_range = ((active_pitches < exp_lo - 12) | (active_pitches > exp_hi + 12))
                    oor_frac = out_of_range.float().mean().item()
                    result["pr_pitch_out_of_range_frac"] = oor_frac
                    if oor_frac > 0.5:
                        result["issues"].append(("label",
                            f"pitch range [{p_lo}-{p_hi}] mostly outside expected [{exp_lo}-{exp_hi}] for {group}"))

    # ── Load and validate conditioning signals ──
    for ckey in ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]:
        cp = cond_paths.get(ckey)
        arr = safe_np(cp) if cp else None
        if arr is None:
            result[f"cond_{ckey}_loaded"] = False
            if cp:
                result["issues"].append(("file", f"conditioning '{ckey}' failed to load"))
            continue
        result[f"cond_{ckey}_loaded"] = True
        t = torch.from_numpy(arr).float().view(-1)
        cond_T = t.shape[0]

        result[f"cond_{ckey}_T"] = cond_T
        result[f"cond_{ckey}_mean"] = t.mean().item()
        result[f"cond_{ckey}_std"] = t.std().item()
        result[f"cond_{ckey}_all_zero"] = bool((t == 0).all())
        result[f"cond_{ckey}_has_nan"] = bool(torch.isnan(t).any())
        result[f"cond_{ckey}_has_inf"] = bool(torch.isinf(t).any())
        result[f"cond_{ckey}_all_nan"] = bool(torch.isnan(t).all())

        # f0 and f0_masked use NaN for unvoiced frames — that's expected
        # Only flag NaN as an issue for non-f0 signals, or if ALL f0 is NaN
        if ckey in ("f0", "f0_masked"):
            if result[f"cond_{ckey}_all_nan"]:
                result["issues"].append(("signal", f"conditioning '{ckey}' is ALL NaN (no voiced frames)"))
        else:
            if result[f"cond_{ckey}_has_nan"]:
                result["issues"].append(("signal", f"conditioning '{ckey}' has NaN"))
        if result[f"cond_{ckey}_has_inf"]:
            result["issues"].append(("signal", f"conditioning '{ckey}' has Inf"))

        # Temporal alignment
        if abs(cond_T - T_slow) > max(5, T_slow * 0.05):
            result["issues"].append(("alignment", f"conditioning '{ckey}' T={cond_T} vs latent T={T_slow}"))

        # Signal-specific checks
        if ckey == "amp":
            result["amp_max"] = t.max().item()
            result["amp_activity"] = (t > 0.01).float().mean().item()
            if result[f"cond_{ckey}_all_zero"]:
                result["issues"].append(("signal", "amplitude is all zero (silent)"))
        elif ckey == "onsets":
            result["onset_count"] = (t > 0.5).sum().item()
            result["onset_density"] = result["onset_count"] / max(1, cond_T) * SLOW_HZ
            if result["onset_count"] == 0 and result.get("pr_activity_ratio", 0) > 0.05:
                result["issues"].append(("consistency", "piano_roll has activity but no onsets detected"))
        elif ckey == "f0":
            voiced_f0 = t[t > 0]
            if voiced_f0.numel() > 0:
                result["f0_median_hz"] = voiced_f0.median().item()
                result["f0_voiced_ratio"] = (t > 0).float().mean().item()
                # Check f0 vs piano_roll pitch consistency
                if result.get("piano_roll_loaded") and not result.get("pr_all_zero", True):
                    f0_midi = 69 + 12 * torch.log2(voiced_f0 / 440.0)
                    f0_midi_med = f0_midi.median().item()
                    pr_lo = result.get("pr_pitch_lo", 0)
                    pr_hi = result.get("pr_pitch_hi", 127)
                    if f0_midi_med < pr_lo - 12 or f0_midi_med > pr_hi + 12:
                        result["issues"].append(("consistency",
                            f"f0 median MIDI {f0_midi_med:.0f} outside piano_roll range [{pr_lo}-{pr_hi}]"))
        elif ckey == "rframe":
            result["rframe_activity"] = (t > 0).float().mean().item()

    # Cross-signal: amp activity vs piano_roll activity
    if result.get("piano_roll_loaded") and not result.get("pr_all_zero", True):
        pr_act = result.get("pr_activity_ratio", 0)
        amp_act = result.get("amp_activity", 0)
        if pr_act > 0.1 and amp_act < 0.01:
            result["issues"].append(("consistency", "piano_roll active but amp is silent"))
        elif amp_act > 0.3 and pr_act < 0.01:
            result["issues"].append(("consistency", "amp active but piano_roll empty"))

    result["issue_count"] = len(result["issues"])
    return result


# ── Aggregate report ─────────────────────────────────────────────────────────

def generate_report(results: List[dict], manifest_path: str) -> str:
    lines = []
    def ln(s=""):
        lines.append(s)

    total = len(results)
    ln(f"{'='*80}")
    ln(f"  MANIFEST QUALITY REPORT")
    ln(f"  Manifest: {manifest_path}")
    ln(f"  Total entries: {total}")
    ln(f"{'='*80}\n")

    # ── 1. File Availability ──
    ln("1. FILE AVAILABILITY")
    ln("-" * 40)
    for ftype in ["latent_path", "encodec_path", "piano_roll_path"]:
        present = sum(1 for r in results if r.get(f"{ftype}_present", False))
        exists = sum(1 for r in results if r.get(f"{ftype}_exists", False))
        loaded_key = ftype.replace("_path", "_loaded")
        loaded = sum(1 for r in results if r.get(loaded_key, False))
        ln(f"  {ftype:25s}: present={present:5d} ({100*present/total:.1f}%)  "
           f"exists={exists:5d} ({100*exists/total:.1f}%)  "
           f"loaded={loaded:5d} ({100*loaded/total:.1f}%)")

    cond_keys = ["onsets", "rframe", "rbend", "amp", "f0", "f0_masked"]
    for ck in cond_keys:
        exists = sum(1 for r in results if r.get(f"cond_{ck}_exists", False))
        loaded = sum(1 for r in results if r.get(f"cond_{ck}_loaded", False))
        ln(f"  cond_{ck:15s}: exists={exists:5d} ({100*exists/total:.1f}%)  "
           f"loaded={loaded:5d} ({100*loaded/total:.1f}%)")

    all_core = sum(1 for r in results
                   if r.get("latent_loaded") and r.get("encodec_loaded") and r.get("piano_roll_loaded"))
    all_full = sum(1 for r in results
                   if r.get("latent_loaded") and r.get("encodec_loaded") and r.get("piano_roll_loaded")
                   and all(r.get(f"cond_{ck}_loaded", False) for ck in cond_keys))
    ln(f"\n  All 3 core streams (latent+encodec+pr): {all_core:5d} ({100*all_core/total:.1f}%)")
    ln(f"  All streams + all conditioning:          {all_full:5d} ({100*all_full/total:.1f}%)")

    # ── 2. Shape Validation ──
    ln(f"\n2. SHAPE VALIDATION")
    ln("-" * 40)
    lat_ok = sum(1 for r in results if r.get("latent_shape_ok", False))
    pr_ok = sum(1 for r in results if r.get("pr_shape_ok", False))
    enc_ok = sum(1 for r in results if r.get("encodec_shape_ok", False))
    loaded_lat = sum(1 for r in results if r.get("latent_loaded", False))
    loaded_pr = sum(1 for r in results if r.get("piano_roll_loaded", False))
    loaded_enc = sum(1 for r in results if r.get("encodec_loaded", False))
    ln(f"  Latent [8,16,T]:    {lat_ok}/{loaded_lat} loaded have correct shape")
    ln(f"  Encodec [<=8,T]:    {enc_ok}/{loaded_enc} loaded have correct shape")
    ln(f"  PianoRoll [128,T]:  {pr_ok}/{loaded_pr} loaded have correct shape")

    # Duration stats
    durations = [r["duration_sec"] for r in results if "duration_sec" in r]
    if durations:
        d = np.array(durations)
        ln(f"\n  Duration stats (sec): min={d.min():.1f}  median={np.median(d):.1f}  "
           f"mean={d.mean():.1f}  max={d.max():.1f}  total_hrs={d.sum()/3600:.1f}")
        short = sum(1 for x in d if x < 2.0)
        if short:
            ln(f"  WARNING: {short} entries shorter than 2 seconds")

    # ── 3. Label Quality ──
    ln(f"\n3. LABEL QUALITY")
    ln("-" * 40)
    group_valid = sum(1 for r in results if r.get("label_group_valid", False))
    sub_valid = sum(1 for r in results if r.get("label_sub_valid", False))
    sub_undef = sum(1 for r in results if r.get("label_sub_undefined", False))
    ln(f"  Valid groups:       {group_valid}/{total} ({100*group_valid/total:.1f}%)")
    ln(f"  Valid subgroups:    {sub_valid}/{total} ({100*sub_valid/total:.1f}%)")
    ln(f"  'undefined' subgroup: {sub_undef}/{total} ({100*sub_undef/total:.1f}%)")

    # Group distribution
    gc = Counter(r["group"] for r in results)
    ln(f"\n  Group distribution:")
    dur_by_group = defaultdict(float)
    for r in results:
        if "duration_sec" in r:
            dur_by_group[r["group"]] += r["duration_sec"]
    for g in APPROVED_GROUPS:
        cnt = gc.get(g, 0)
        hrs = dur_by_group.get(g, 0) / 3600
        ln(f"    {g:12s}: {cnt:6d} entries ({100*cnt/total:5.1f}%)  {hrs:6.1f} hrs")

    # Subgroup distribution
    sc = Counter(r["sub_group"] for r in results)
    ln(f"\n  Subgroup distribution:")
    for sg, cnt in sc.most_common():
        ln(f"    {sg:20s}: {cnt:6d} ({100*cnt/total:5.1f}%)")

    # Subgroup-group mismatches
    mismatches = [(r["idx"], r["group"], r["sub_group"]) for r in results if not r.get("label_sub_valid", True)]
    if mismatches:
        mismatch_counter = Counter((g, sg) for _, g, sg in mismatches)
        ln(f"\n  Group↔Subgroup mismatches: {len(mismatches)} entries")
        for (g, sg), cnt in mismatch_counter.most_common(10):
            ln(f"    {g}/{sg}: {cnt}")

    # ── Undefined subgroup per group ──
    ln(f"\n  'undefined' subgroup rate per group:")
    for g in APPROVED_GROUPS:
        g_total = gc.get(g, 0)
        g_undef = sum(1 for r in results if r["group"] == g and r.get("label_sub_undefined"))
        if g_total > 0:
            ln(f"    {g:12s}: {g_undef}/{g_total} ({100*g_undef/g_total:.1f}%)")

    # ── 4. Signal Quality ──
    ln(f"\n4. SIGNAL QUALITY")
    ln("-" * 40)

    # Latent quality
    lat_nan = sum(1 for r in results if r.get("latent_has_nan"))
    lat_inf = sum(1 for r in results if r.get("latent_has_inf"))
    lat_zero = sum(1 for r in results if r.get("latent_all_zero"))
    lat_stds = [r["latent_std"] for r in results if "latent_std" in r]
    lat_low_var = sum(1 for s in lat_stds if s < 0.01)
    ln(f"  Latents:")
    ln(f"    NaN: {lat_nan}  Inf: {lat_inf}  All-zero: {lat_zero}  Low-variance(<0.01): {lat_low_var}")
    if lat_stds:
        s = np.array(lat_stds)
        ln(f"    Std stats: min={s.min():.4f}  median={np.median(s):.4f}  mean={s.mean():.4f}  max={s.max():.4f}")

    # Encodec quality
    enc_zero = sum(1 for r in results if r.get("encodec_all_zero"))
    ln(f"  Encodec:")
    ln(f"    All-zero tokens: {enc_zero}")

    # Piano roll quality
    pr_zero = sum(1 for r in results if r.get("pr_all_zero"))
    pr_sparse = sum(1 for r in results if r.get("pr_activity_ratio", 1) < 0.02 and not r.get("pr_all_zero", True))
    polys = [r["pr_mean_polyphony"] for r in results if "pr_mean_polyphony" in r and r["pr_mean_polyphony"] > 0]
    ln(f"  Piano Roll:")
    ln(f"    All-zero: {pr_zero}  Very sparse (<2% active): {pr_sparse}")
    if polys:
        p = np.array(polys)
        ln(f"    Polyphony stats: min={p.min():.1f}  median={np.median(p):.1f}  mean={p.mean():.1f}  max={p.max():.1f}")

    # Pitch range violations per group
    pitch_issues = sum(1 for r in results if r.get("pr_pitch_out_of_range_frac", 0) > 0.5)
    if pitch_issues:
        ln(f"    Pitch range violations (>50% out of range): {pitch_issues}")
        for g in APPROVED_GROUPS:
            g_violations = sum(1 for r in results
                               if r["group"] == g and r.get("pr_pitch_out_of_range_frac", 0) > 0.5)
            if g_violations:
                g_total = gc.get(g, 0)
                ln(f"      {g}: {g_violations}/{g_total}")

    # Conditioning quality
    ln(f"  Conditioning signals:")
    for ck in cond_keys:
        zeros = sum(1 for r in results if r.get(f"cond_{ck}_all_zero"))
        nans = sum(1 for r in results if r.get(f"cond_{ck}_has_nan"))
        all_nans = sum(1 for r in results if r.get(f"cond_{ck}_all_nan"))
        infs = sum(1 for r in results if r.get(f"cond_{ck}_has_inf"))
        loaded = sum(1 for r in results if r.get(f"cond_{ck}_loaded"))
        extra = ""
        if ck in ("f0", "f0_masked") and nans > 0:
            extra = f"  (NaN expected for unvoiced; all_NaN={all_nans})"
        ln(f"    {ck:15s}: loaded={loaded:5d}  all_zero={zeros:5d}  NaN={nans}  Inf={infs}{extra}")

    # Onset density
    onset_densities = [r["onset_density"] for r in results if "onset_density" in r and r["onset_density"] > 0]
    if onset_densities:
        od = np.array(onset_densities)
        ln(f"    Onset density (onsets/sec): min={od.min():.2f}  median={np.median(od):.2f}  "
           f"mean={od.mean():.2f}  max={od.max():.2f}")

    # F0 stats
    f0_meds = [r["f0_median_hz"] for r in results if "f0_median_hz" in r]
    if f0_meds:
        f = np.array(f0_meds)
        ln(f"    F0 median (Hz): min={f.min():.0f}  median={np.median(f):.0f}  mean={f.mean():.0f}  max={f.max():.0f}")

    # ── 5. Cross-Signal Consistency ──
    ln(f"\n5. CROSS-SIGNAL CONSISTENCY")
    ln("-" * 40)

    # Ensure issue_count is set on all results
    for r in results:
        if "issue_count" not in r:
            r["issue_count"] = len(r.get("issues", []))

    issue_types = defaultdict(int)
    for r in results:
        for cat, msg in r.get("issues", []):
            issue_types[(cat, msg)] += 1

    consistency_issues = {k: v for k, v in issue_types.items() if k[0] == "consistency"}
    if consistency_issues:
        for (cat, msg), cnt in sorted(consistency_issues.items(), key=lambda x: -x[1]):
            ln(f"    {msg}: {cnt} entries ({100*cnt/total:.1f}%)")
    else:
        ln("  No consistency issues detected.")

    alignment_issues = {k: v for k, v in issue_types.items() if k[0] == "alignment"}
    if alignment_issues:
        ln(f"\n  Temporal alignment issues:")
        for (cat, msg), cnt in sorted(alignment_issues.items(), key=lambda x: -x[1]):
            ln(f"    {msg}: {cnt} entries")

    # ── 6. Per-Group Quality Breakdown ──
    ln(f"\n6. PER-GROUP QUALITY BREAKDOWN")
    ln("-" * 40)
    for g in APPROVED_GROUPS:
        g_results = [r for r in results if r["group"] == g]
        if not g_results:
            continue
        g_total = len(g_results)
        g_clean = sum(1 for r in g_results if r["issue_count"] == 0)
        g_issues = sum(r["issue_count"] for r in g_results)

        ln(f"\n  [{g.upper()}] ({g_total} entries)")
        ln(f"    Clean entries (0 issues): {g_clean} ({100*g_clean/g_total:.1f}%)")
        ln(f"    Total issues: {g_issues} (avg {g_issues/g_total:.2f} per entry)")

        # Top issues for this group
        g_issue_counter = Counter()
        for r in g_results:
            for cat, msg in r.get("issues", []):
                g_issue_counter[(cat, msg)] += 1
        if g_issue_counter:
            ln(f"    Top issues:")
            for (cat, msg), cnt in g_issue_counter.most_common(5):
                ln(f"      [{cat}] {msg}: {cnt}")

        # Subgroup breakdown within group
        g_subs = Counter(r["sub_group"] for r in g_results)
        ln(f"    Subgroups: {dict(g_subs.most_common())}")

    # ── 7. Issue Summary ──
    ln(f"\n7. ISSUE SUMMARY (TOP PROBLEMS)")
    ln("-" * 40)
    all_issues_flat = Counter()
    cat_counter = Counter()
    for r in results:
        for cat, msg in r.get("issues", []):
            all_issues_flat[msg] += 1
            cat_counter[cat] += 1

    clean = sum(1 for r in results if r["issue_count"] == 0)
    has_issues = total - clean
    ln(f"  Clean entries: {clean}/{total} ({100*clean/total:.1f}%)")
    ln(f"  Entries with issues: {has_issues}/{total} ({100*has_issues/total:.1f}%)")
    ln(f"\n  Issues by category:")
    for cat, cnt in cat_counter.most_common():
        ln(f"    {cat:15s}: {cnt:6d} occurrences")
    ln(f"\n  Top 20 specific issues:")
    for msg, cnt in all_issues_flat.most_common(20):
        ln(f"    ({cnt:5d}) {msg}")

    # ── 8. Overall Quality Score ──
    ln(f"\n8. OVERALL QUALITY SCORE")
    ln("=" * 40)

    # Scoring: 0-100 based on weighted factors
    scores = {}

    # File completeness (25 pts)
    file_score = 25 * (all_core / total) if total > 0 else 0
    scores["File completeness (all core streams)"] = (file_score, 25)

    # Label validity (15 pts)
    label_score = 15 * (group_valid / total) * (sub_valid / total) if total > 0 else 0
    # Penalize high undefined rate
    undef_penalty = min(1.0, sub_undef / total / 0.5)  # 50%+ undefined = max penalty
    label_score *= (1 - 0.3 * undef_penalty)
    scores["Label quality (group/subgroup)"] = (label_score, 15)

    # Signal integrity (20 pts) - NaN, Inf, all-zero
    bad_signals = lat_nan + lat_inf + lat_zero + enc_zero + pr_zero
    # For f0/f0_masked, only count all-NaN as a problem (partial NaN is expected)
    bad_cond = sum(1 for r in results for ck in cond_keys
                   if r.get(f"cond_{ck}_all_zero")
                   or (ck not in ("f0", "f0_masked") and r.get(f"cond_{ck}_has_nan"))
                   or (ck in ("f0", "f0_masked") and r.get(f"cond_{ck}_all_nan")))
    signal_problems = (bad_signals + bad_cond) / max(1, total * (3 + len(cond_keys)))
    signal_score = 20 * (1 - min(1.0, signal_problems * 5))
    scores["Signal integrity (no NaN/zero)"] = (signal_score, 20)

    # Data balance (10 pts) - group distribution evenness
    if gc:
        counts = [gc.get(g, 0) for g in APPROVED_GROUPS]
        ideal = total / len(APPROVED_GROUPS)
        imbalance = sum(abs(c - ideal) for c in counts) / (2 * total)
        balance_score = 10 * (1 - min(1.0, imbalance * 2))
    else:
        balance_score = 0
    scores["Data balance (group distribution)"] = (balance_score, 10)

    # Cross-signal consistency (15 pts)
    consistency_count = sum(v for k, v in issue_types.items() if k[0] in ("consistency", "alignment"))
    consistency_rate = consistency_count / max(1, total)
    consistency_score = 15 * (1 - min(1.0, consistency_rate * 3))
    scores["Cross-signal consistency"] = (consistency_score, 15)

    # Pitch/label accuracy (15 pts)
    pitch_violation_rate = pitch_issues / max(1, total)
    mismatch_rate = len(mismatches) / max(1, total)
    pitch_label_score = 15 * (1 - min(1.0, (pitch_violation_rate + mismatch_rate) * 3))
    scores["Pitch/label accuracy"] = (pitch_label_score, 15)

    total_score = sum(s for s, _ in scores.values())
    max_score = sum(m for _, m in scores.values())

    for name, (s, m) in scores.items():
        bar = "#" * int(s / m * 20) + "." * (20 - int(s / m * 20))
        ln(f"  {name:42s}: {s:5.1f}/{m:2d}  [{bar}]")
    ln(f"\n  {'TOTAL':42s}: {total_score:5.1f}/{max_score}")

    grade = "A+" if total_score >= 95 else "A" if total_score >= 90 else "B" if total_score >= 80 else \
            "C" if total_score >= 70 else "D" if total_score >= 60 else "F"
    ln(f"  Grade: {grade}")

    # Training readiness assessment
    ln(f"\n9. TRAINING READINESS ASSESSMENT")
    ln("=" * 40)
    blockers = []
    warnings = []
    if all_core / total < 0.8:
        blockers.append(f"Only {100*all_core/total:.0f}% entries have all core streams")
    if lat_nan + lat_inf > 0:
        blockers.append(f"{lat_nan+lat_inf} entries have NaN/Inf in latents")
    if group_valid / total < 0.95:
        blockers.append(f"{total-group_valid} entries have invalid group labels")

    if sub_undef / total > 0.3:
        warnings.append(f"{100*sub_undef/total:.0f}% subgroups are 'undefined' - model can't learn fine instrument distinctions")
    if pr_zero > total * 0.05:
        warnings.append(f"{pr_zero} entries ({100*pr_zero/total:.1f}%) have empty piano rolls - no pitch guidance")
    if enc_zero > total * 0.05:
        warnings.append(f"{enc_zero} entries have all-zero encodec tokens - no timbre info")
    if pitch_issues > total * 0.02:
        warnings.append(f"{pitch_issues} entries have pitch ranges wrong for their labeled group")
    if len(mismatches) > total * 0.02:
        warnings.append(f"{len(mismatches)} entries have subgroup↔group mismatches")

    min_group = min(gc.values()) if gc else 0
    max_group = max(gc.values()) if gc else 0
    if max_group > 4 * min_group and min_group > 0:
        warnings.append(f"Group imbalance: largest={max_group}, smallest={min_group} ({max_group/min_group:.1f}x ratio)")

    if blockers:
        ln("  BLOCKERS (fix before training):")
        for b in blockers:
            ln(f"    [!] {b}")
    else:
        ln("  No blockers found.")

    if warnings:
        ln("  WARNINGS (may affect quality):")
        for w in warnings:
            ln(f"    [~] {w}")
    else:
        ln("  No warnings.")

    ln(f"\n{'='*80}")
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Manifest Quality Estimator")
    parser.add_argument("manifest", help="Path to manifest JSON file")
    parser.add_argument("--sample", type=int, default=0,
                        help="Analyze only N random entries (0=all)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers")
    parser.add_argument("--remap", action="store_true", default=True,
                        help="Remap /mnt/msdd paths to bucket paths (default: true)")
    parser.add_argument("--no-remap", dest="remap", action="store_false",
                        help="Don't remap paths (use as-is)")
    parser.add_argument("--output", type=str, default=None,
                        help="Write report to file (default: stdout)")
    args = parser.parse_args()

    print(f"Loading manifest: {args.manifest}")
    with open(args.manifest, "rb") as f:
        manifest = orjson.loads(f.read())
    print(f"  {len(manifest)} entries loaded")

    if args.remap:
        print("  Remapping /mnt/msdd paths to bucket paths...")
        manifest = [remap_entry(e) for e in manifest]

    if args.sample > 0 and args.sample < len(manifest):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(manifest), size=args.sample, replace=False)
        manifest = [manifest[i] for i in sorted(indices)]
        print(f"  Sampled {len(manifest)} entries")

    print(f"  Analyzing with {args.workers} workers...")
    t0 = time.time()

    results = []
    if args.workers <= 1:
        for i, entry in enumerate(manifest):
            if i % 500 == 0:
                print(f"    {i}/{len(manifest)}...")
            try:
                results.append(analyze_entry(entry, i))
            except Exception as e:
                results.append({"idx": i, "group": entry.get("group", "?"),
                                "sub_group": entry.get("sub_group", "?"),
                                "issues": [("error", str(e))], "issue_count": 1})
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(analyze_entry, entry, i): i
                       for i, entry in enumerate(manifest)}
            done = 0
            for fut in as_completed(futures):
                done += 1
                if done % 500 == 0:
                    print(f"    {done}/{len(manifest)}...")
                try:
                    results.append(fut.result())
                except Exception as e:
                    idx = futures[fut]
                    results.append({"idx": idx, "group": "?", "sub_group": "?",
                                    "issues": [("error", str(e))], "issue_count": 1})

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(manifest)/elapsed:.0f} entries/sec)")

    report = generate_report(results, args.manifest)
    if args.output:
        Path(args.output).write_text(report)
        print(f"  Report written to {args.output}")
    else:
        print("\n" + report)


if __name__ == "__main__":
    main()
