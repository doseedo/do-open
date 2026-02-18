#!/usr/bin/env python3
"""
Visual alignment checker (with latent-key selection & structure dump)

Now also analyzes ONSETS:
- Loads conditioning_paths['onsets'] if present (expects 1D per slow-frame; float strength or 0/1)
- Plots raw onset strength and detected impulses (via z-score threshold + minimum distance)
- Includes SNAPPED versions and correlation/lag checks vs latent & amp
"""

import argparse, json, os, random
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------- Grids ----------
DCAE_SR, DCAE_HOP = 44100, 4096    # slow grid
ENC_SR,  ENC_HOP  = 24000, 320     # fast grid

# ---------- Time helpers ----------
def sec_per_slow_frame() -> float: return DCAE_HOP / float(DCAE_SR)
def sec_per_fast_frame() -> float: return ENC_HOP  / float(ENC_SR)
def expected_fast_from_slow(T_slow: int) -> int:
    return int(round(T_slow * ( (ENC_SR/ENC_HOP) / (DCAE_SR/DCAE_HOP) )))

# ---------- Safe loaders ----------
def safe_np_load(path: Optional[Path]) -> np.ndarray:
    if path is None: raise FileNotFoundError("None path")
    if not path.exists() or not path.is_file(): raise FileNotFoundError(str(path))
    try: return np.load(path)
    except ValueError: return np.load(path, allow_pickle=True)

def load_encodec_tokens(encodec_path: Path) -> np.ndarray:
    blob = torch.load(encodec_path, map_location="cpu")
    t = first_tensor(blob)
    if t is None: raise RuntimeError(f"No encodec tensor in {encodec_path}")
    x = t.detach().cpu().numpy()
    if x.ndim == 1: x = x[None, :]
    elif x.ndim == 3: x = x[0]
    return x  # [C, T]

# ---------- Latent tensor discovery ----------
def describe_structure(obj: Any, prefix: str="") -> List[str]:
    lines: List[str] = []
    def rec(x, pfx):
        if isinstance(x, torch.Tensor):
            lines.append(f"{pfx} :: Tensor {tuple(x.shape)} dtype={x.dtype}")
            return
        if isinstance(x, dict):
            for k,v in x.items():
                rec(v, f"{pfx}/{k}" if pfx else str(k))
        elif isinstance(x, (list, tuple)):
            for i,v in enumerate(x):
                rec(v, f"{pfx}/{i}" if pfx else str(i))
    rec(obj, prefix)
    return lines

def get_by_keypath(obj: Any, keypath: Optional[str]) -> Any:
    if keypath in (None, "", "."): return obj
    cur = obj
    for tok in keypath.split("/"):
        if tok == "": continue
        if isinstance(cur, dict):
            if tok not in cur: raise KeyError(f"Missing key '{tok}' in dict at path '{keypath}'")
            cur = cur[tok]
        elif isinstance(cur, (list, tuple)):
            try: idx = int(tok)
            except: raise KeyError(f"Non-integer index '{tok}' into list at '{keypath}'")
            if not (0 <= idx < len(cur)): raise KeyError(f"Index {idx} out of range at '{keypath}'")
            cur = cur[idx]
        else:
            raise KeyError(f"Path '{keypath}' hits non-container at '{tok}'")
    return cur

def first_tensor(x: Any) -> Optional[torch.Tensor]:
    if isinstance(x, torch.Tensor): return x
    if isinstance(x, dict):
        for k in ("latents","z","codes","tokens","encodec","audio_tokens","data"):
            if k in x:
                t = first_tensor(x[k])
                if t is not None: return t
        for v in x.values():
            t = first_tensor(v)
            if t is not None: return t
    if isinstance(x, (list,tuple)):
        for v in x:
            t = first_tensor(v)
            if t is not None: return t
    return None

# ---------- Latent handling ----------
def latent_to_time_last(x: torch.Tensor, force_time_axis: Optional[int], expected_sec: float) -> Tuple[int, torch.Tensor]:
    if x.ndim == 1:
        y = x.unsqueeze(0)  # [1, T]
        return int(y.shape[-1]), y
    candidates = []
    axes_to_try = [force_time_axis] if force_time_axis is not None else [-1, -2]
    tried = set()
    for ax in axes_to_try:
        if ax in tried or ax is None: continue
        tried.add(ax)
        if not (-x.ndim <= ax < x.ndim): continue
        y = x.moveaxis(ax, -1)
        T = int(y.shape[-1]); sec = T * sec_per_slow_frame()
        score = abs(sec - expected_sec)
        candidates.append((score, T, y, ax))
    if not candidates:
        y, T = x, int(x.shape[-1])
        return T, y
    candidates.sort(key=lambda t: t[0])
    _, T, y, ax = candidates[0]
    print(f"[DEBUG] latent time axis chosen = {ax} (T={T})")
    return T, y

def reduce_latent(y_time_last: torch.Tensor, agg: str) -> np.ndarray:
    T = int(y_time_last.shape[-1])
    Y = y_time_last.reshape(-1, T)  # [F, T]
    if agg == "l2":
        out = torch.linalg.vector_norm(Y, dim=0)
    elif agg == "l1":
        out = torch.sum(torch.abs(Y), dim=0)
    elif agg == "mean":
        out = torch.mean(Y, dim=0)
    elif agg == "pc1":
        Yc = Y - Y.mean(dim=1, keepdim=True)
        U, S, Vh = torch.linalg.svd(Yc, full_matrices=False)
        out = Vh[0]
        try:
            energy = torch.sum(Y**2, dim=0)
            corr = torch.corrcoef(torch.stack([out, energy]))[0,1]
            if torch.isfinite(corr) and corr < 0: out = -out
        except Exception:
            pass
    else:
        raise ValueError(f"Unknown --latent-agg '{agg}'")
    return out.detach().cpu().numpy().astype(np.float32)

# ---------- crop/pad ----------
def crop_or_pad_1d(a: np.ndarray, T: int, v: float=0.0) -> np.ndarray:
    t = a.shape[-1]
    if t == T: return a
    if t > T:  return a[..., :T]
    pad = np.full(a.shape[:-1] + (T-t,), v, dtype=a.dtype)
    return np.concatenate([a, pad], axis=-1)

def crop_or_pad_2d(a: np.ndarray, T: int, v: float=0.0) -> np.ndarray:
    F, t = a.shape[0], a.shape[-1]
    if t == T: return a
    if t > T:  return a[..., :T]
    pad = np.full((F, T-t), v, dtype=a.dtype)
    return np.concatenate([a, pad], axis=-1)

# ---------- plots ----------
def plot_series(t: np.ndarray, y: np.ndarray, title: str, ylabel: str):
    plt.figure(); plt.plot(t, y); plt.xlabel("Seconds"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout()

def plot_binary(t: np.ndarray, y: np.ndarray, title: str, ylabel: str):
    plt.figure(); plt.step(t, y, where="post"); plt.xlabel("Seconds"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout()

def plot_heatmap(t: np.ndarray, M: np.ndarray, title: str, ylabel: str):
    plt.figure()
    extent = [t[0], t[-1] if len(t)>1 else 0.0, 0, M.shape[0]]
    plt.imshow(M, aspect="auto", origin="lower", extent=extent)
    plt.xlabel("Seconds"); plt.ylabel(ylabel); plt.title(title); plt.tight_layout()

def plot_overlay(t: np.ndarray, a: np.ndarray, b: np.ndarray, title: str, la="latent_norm (z)", lb="RMS/onset (z)"):
    def z(x): x = np.asarray(x); return (x - x.mean()) / (x.std()+1e-8)
    L = min(len(a), len(b)); A, B = z(a[:L]), z(b[:L])
    plt.figure(); plt.plot(t[:L], A, label=la); plt.plot(t[:L], B, label=lb, alpha=0.75)
    plt.legend(); plt.xlabel("Seconds"); plt.ylabel("z-score"); plt.title(title); plt.tight_layout()

def corr_lag(a: np.ndarray, b: np.ndarray) -> Tuple[float,int,float]:
    L = min(len(a), len(b))
    if L == 0: return float("nan"),0,float("nan")
    def z(x): x = (x - x.mean()) / (x.std()+1e-8); return x
    A,B = z(a[:L]), z(b[:L])
    r0 = float(np.corrcoef(A,B)[0,1])
    xcorr = np.correlate(A,B,mode="full")
    lags = np.arange(-L+1, L); best = int(lags[int(xcorr.argmax())])
    return r0, best, best*sec_per_slow_frame()

# ---------- onset helpers ----------
def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    mu, sd = x.mean(), x.std()
    return (x - mu) / (sd + 1e-8)

def pick_peaks(x: np.ndarray, z_thresh: float=2.5, min_dist_frames: int=3) -> np.ndarray:
    """
    Simple peak picking on z-scored series. Returns binary 0/1 array.
    - z_thresh: threshold on z-score
    - min_dist_frames: minimum distance between picked peaks (post-suppression)
    """
    xz = zscore(x)
    cand = (xz >= z_thresh).astype(np.int32)
    # Non-maximum suppression over local window
    peaks = np.zeros_like(cand)
    i = 0
    T = len(xz)
    while i < T:
        if cand[i]:
            # look ahead window to find local max index
            j_end = min(T, i + min_dist_frames + 1)
            j = i
            # extend while within min_dist AND candidates
            # choose the argmax of z within [i, j_end)
            local_slice = xz[i:j_end]
            k = int(np.argmax(local_slice))
            peak_idx = i + k
            peaks[peak_idx] = 1
            i = peak_idx + min_dist_frames + 1
        else:
            i += 1
    return peaks

# ---------- entry selection ----------
def pick_entry(data: list, idx: Optional[int], seed: Optional[int]) -> Dict[str,Any]:
    if idx is not None:
        if not (0 <= idx < len(data)): raise IndexError(f"--index out of range 0..{len(data)-1}")
        return data[idx]
    rng = random.Random(seed); return rng.choice(data)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("final_training_manifest.json"))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--save-dir", type=Path, default=None)
    ap.add_argument("--snap", action="store_true")
    ap.add_argument("--print-structure", action="store_true", help="Print tensor tree of latent .pt")
    ap.add_argument("--latent-key", type=str, default=None, help="Slash path into latent file (e.g., 'latents' or 'latents/z')")
    ap.add_argument("--time-axis", type=int, default=None, help="Force latent time axis (e.g., -1 or -2)")
    ap.add_argument("--latent-agg", type=str, default="l2", choices=["l2","l1","mean","pc1"], help="Per-frame reduction")
    # Onset analysis params
    ap.add_argument("--onset-z", type=float, default=2.5, help="z-score threshold for peak picking")
    ap.add_argument("--onset-min-dist", type=int, default=3, help="min distance between peaks (slow frames)")
    args = ap.parse_args()

    if not args.manifest.exists(): raise FileNotFoundError(f"Manifest not found: {args.manifest}")
    data = json.loads(args.manifest.read_text())
    entry = pick_entry(data, args.index, args.seed)

    audio_path  = entry.get("audio_path","<unknown>")
    latent_path = Path(entry["latent_path"])
    pr_path     = Path(entry["piano_roll_path"])
    encodec_path= Path(entry["encodec_path"])
    cond        = entry.get("conditioning_paths") or {}
    amp_p    = Path(cond["amp"])     if isinstance(cond.get("amp"), str)     and Path(cond["amp"]).is_file()     else None
    rframe_p = Path(cond["rframe"])  if isinstance(cond.get("rframe"), str)  and Path(cond["rframe"]).is_file()  else None
    rbend_p  = Path(cond["rbend"])   if isinstance(cond.get("rbend"), str)   and Path(cond["rbend"]).is_file()   else None
    onsets_p = Path(cond["onsets"])  if isinstance(cond.get("onsets"), str)  and Path(cond["onsets"]).is_file()  else None

    print(f"\n▶ Entry: {audio_path}\n   latent:  {latent_path}\n   pr:      {pr_path}\n   encodec: {encodec_path}\n   amp:     {amp_p}\n   rframe:  {rframe_p}\n   rbend:   {rbend_p}\n   onsets:  {onsets_p}")

    # Encodec first
    enc = load_encodec_tokens(encodec_path)       # [C, T_fast]
    T_fast = enc.shape[-1]; enc_sec = T_fast*sec_per_fast_frame()

    # Latents: open + structure dump
    blob = torch.load(latent_path, map_location="cpu")
    if args.print_structure:
        print("\n--- latent .pt structure ---")
        for line in describe_structure(blob):
            print(line)
        print("----------------------------\n")

    # Choose tensor via key-path (or auto-first)
    try:
        chosen = get_by_keypath(blob, args.latent_key)
    except Exception as e:
        raise RuntimeError(f"--latent-key failed: {e}")
    t = chosen if isinstance(chosen, torch.Tensor) else first_tensor(chosen)
    if t is None: raise RuntimeError("No tensor found at the chosen path")

    # Time axis detection/forcing
    T_lat, y_time_last = latent_to_time_last(t.detach().float(), args.time_axis, enc_sec)

    # Per-frame reduction
    lat_series = reduce_latent(y_time_last, args.latent_agg)   # np[ T_lat ]

    # Piano roll & conditioning
    pr = safe_np_load(pr_path); 
    if pr.ndim != 2: raise RuntimeError(f"Piano roll unexpected shape: {pr.shape}")
    T_pr = pr.shape[-1]
    amp    = safe_np_load(amp_p)    if amp_p    else None
    rframe = safe_np_load(rframe_p) if rframe_p else None
    rbend  = safe_np_load(rbend_p)  if rbend_p  else None
    onsets_raw = safe_np_load(onsets_p) if onsets_p else None

    # Sanitize onsets shape: allow (T,) or (1,T) or (F,T)->take first row
    if isinstance(onsets_raw, np.ndarray):
        if onsets_raw.ndim == 2 and onsets_raw.shape[0] == 1:
            onsets_raw = onsets_raw[0]
        elif onsets_raw.ndim == 2 and onsets_raw.shape[0] > 1:
            # Heuristic: take max across features if multi-band onsets provided
            onsets_raw = onsets_raw.max(axis=0)
        elif onsets_raw.ndim != 1:
            raise RuntimeError(f"Onsets unexpected shape: {onsets_raw.shape}")

    # Derived: picked onset impulses
    onsets_imp = None
    if isinstance(onsets_raw, np.ndarray):
        onsets_imp = pick_peaks(onsets_raw.astype(np.float32), z_thresh=args.onset_z, min_dist_frames=args.onset_min_dist)

    # A generic "slow" length to print (prefer amp, else onsets, else rframe/rbend)
    slow_lengths = [a.shape[-1] for a in [amp, onsets_raw, rframe, rbend] if isinstance(a, np.ndarray)]
    T_cond_repr = slow_lengths[0] if slow_lengths else None

    # Diagnostics
    exp_fast_from_lat = expected_fast_from_slow(T_lat)
    print(f"\n[INFO] Slow frames: latent={T_lat}, cond_repr={T_cond_repr}, pr={T_pr}")
    print(f"[INFO] Fast frames:  encodec={T_fast}, expected≈{exp_fast_from_lat}")
    print(f"[INFO] Durations:    slow≈{T_lat*sec_per_slow_frame():.2f}s, fast≈{T_fast*sec_per_fast_frame():.2f}s")
    print(f"[INFO] latent-agg:   {args.latent_agg}    latent-key: {args.latent_key or '<auto-first>'}")

    # RAW plots
    t_lat  = np.arange(T_lat)*sec_per_slow_frame()
    t_pr   = np.arange(T_pr)*sec_per_slow_frame()
    t_fast = np.arange(T_fast)*sec_per_fast_frame()

    plot_series(t_lat, lat_series, "Latent per-frame summary (RAW)", f"{args.latent_agg}")
    if isinstance(amp, np.ndarray):
        t_amp = np.arange(amp.shape[-1])*sec_per_slow_frame()
        plot_series(t_amp, amp, "Conditioning: amplitude (RAW)", "RMS")
    if isinstance(rframe, np.ndarray):
        t_rfr = np.arange(rframe.shape[-1])*sec_per_slow_frame()
        plot_binary(t_rfr, rframe, "Conditioning: rframe voiced (RAW)", "voiced")
    if isinstance(rbend, np.ndarray):
        t_rbd = np.arange(rbend.shape[-1])*sec_per_slow_frame()
        plot_series(t_rbd, rbend, "Conditioning: rbend semitones (RAW)", "semitones")
    if isinstance(onsets_raw, np.ndarray):
        t_on  = np.arange(onsets_raw.shape[-1])*sec_per_slow_frame()
        plot_series(t_on, onsets_raw, "Conditioning: onsets strength (RAW)", "strength")
        if isinstance(onsets_imp, np.ndarray):
            plot_binary(t_on, onsets_imp, f"Conditioning: onsets (picked) RAW  (z≥{args.onset_z}, min_dist={args.onset_min_dist})", "impulse")

    plot_heatmap(t_pr, pr, "Piano roll (RAW)", "pitch bins")
    plot_heatmap(t_fast, enc, "Encodec tokens (RAW)", "codebook")

    # Correlations (RAW)
    if isinstance(amp, np.ndarray):
        r0, lag, lag_s = corr_lag(lat_series, amp)
        print(f"[CHECK RAW] latent↔RMS      corr@lag0={r0:.3f}, best_lag={lag} frames (~{lag_s:.3f}s)")
    if isinstance(onsets_raw, np.ndarray):
        r0o, lago, lago_s = corr_lag(lat_series, onsets_raw)
        print(f"[CHECK RAW] latent↔onsetStr corr@lag0={r0o:.3f}, best_lag={lago} frames (~{lago_s:.3f}s)")
        if isinstance(amp, np.ndarray):
            r0oa, lagoa, lagoa_s = corr_lag(onsets_raw, amp)
            print(f"[CHECK RAW] onsetStr↔RMS   corr@lag0={r0oa:.3f}, best_lag={lagoa} frames (~{lagoa_s:.3f}s)")

    # SNAP
    if args.snap:
        # Use the common ref length across slow-grid series we actually have
        slow_series = [lat_series]
        if isinstance(amp, np.ndarray): slow_series.append(amp)
        if isinstance(rframe, np.ndarray): slow_series.append(rframe.astype(np.float32))
        if isinstance(rbend, np.ndarray): slow_series.append(rbend)
        if isinstance(onsets_raw, np.ndarray): slow_series.append(onsets_raw)
        L_ref = min(s.shape[-1] for s in slow_series)

        lat_s  = crop_or_pad_1d(lat_series, L_ref, 0.0)
        amp_s  = crop_or_pad_1d(amp, L_ref, 0.0) if isinstance(amp,np.ndarray) else None
        rfr_s  = crop_or_pad_1d(rframe, L_ref, 0.0) if isinstance(rframe,np.ndarray) else None
        rbd_s  = crop_or_pad_1d(rbend,  L_ref, 0.0) if isinstance(rbend,np.ndarray)  else None
        on_s   = crop_or_pad_1d(onsets_raw, L_ref, 0.0) if isinstance(onsets_raw,np.ndarray) else None
        on_imp_s = crop_or_pad_1d(onsets_imp, L_ref, 0.0) if isinstance(onsets_imp,np.ndarray) else None

        pr_s   = crop_or_pad_2d(pr, L_ref, 0.0)
        E_tgt  = expected_fast_from_slow(L_ref)
        enc_s  = crop_or_pad_2d(enc, E_tgt, 0.0)

        t_slow = np.arange(L_ref)*sec_per_slow_frame()
        t_fast_s = np.arange(E_tgt)*sec_per_fast_frame()

        plot_series(t_slow, lat_s, f"Latent per-frame summary (SNAPPED, {args.latent_agg})", f"{args.latent_agg}")
        if amp_s is not None:   plot_series(t_slow, amp_s, "Conditioning: amplitude (SNAPPED)", "RMS")
        if rfr_s is not None:   plot_binary(t_slow, rfr_s, "Conditioning: rframe voiced (SNAPPED)", "voiced")
        if rbd_s is not None:   plot_series(t_slow, rbd_s, "Conditioning: rbend semitones (SNAPPED)", "semitones")
        if on_s is not None:
            plot_series(t_slow, on_s, "Conditioning: onsets strength (SNAPPED)", "strength")
            if on_imp_s is not None:
                plot_binary(t_slow, on_imp_s, f"Conditioning: onsets (picked) SNAPPED (z≥{args.onset_z}, min_dist={args.onset_min_dist})", "impulse")

        plot_heatmap(t_slow, pr_s, "Piano roll (SNAPPED)", "pitch bins")
        plot_heatmap(t_fast_s, enc_s, "Encodec tokens (SNAPPED)", "codebook")

        # Overlays & correlations
        if amp_s is not None:
            plot_overlay(t_slow, lat_s, amp_s, "Latent vs RMS (SNAPPED, z-scored)", lb="RMS (z)")
            r0s, lag_slow, lag_sec = corr_lag(lat_s, amp_s)
            print(f"[CHECK SNAP] latent↔RMS      corr@lag0={r0s:.3f}, best_lag={lag_slow} frames (~{lag_sec:.3f}s)")
        if on_s is not None:
            plot_overlay(t_slow, lat_s, on_s,  "Latent vs onsetStrength (SNAPPED, z-scored)", lb="onsetStr (z)")
            r0so, lag_slow_o, lag_sec_o = corr_lag(lat_s, on_s)
            print(f"[CHECK SNAP] latent↔onsetStr corr@lag0={r0so:.3f}, best_lag={lag_slow_o} frames (~{lag_sec_o:.3f}s)")
            if amp_s is not None:
                plot_overlay(t_slow, on_s, amp_s, "onsetStrength vs RMS (SNAPPED, z-scored)", la="onsetStr (z)", lb="RMS (z)")
                r0soa, lag_slow_oa, lag_sec_oa = corr_lag(on_s, amp_s)
                print(f"[CHECK SNAP] onsetStr↔RMS   corr@lag0={r0soa:.3f}, best_lag={lag_slow_oa} frames (~{lag_sec_oa:.3f}s)")

        print(f"[SNAP] L_ref={L_ref} frames ({L_ref*sec_per_slow_frame():.2f}s), E_target={E_tgt} tokens ({E_tgt*sec_per_fast_frame():.2f}s)")

    # Save or show
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for i, num in enumerate(plt.get_fignums(), start=1):
            plt.figure(num).savefig(args.save_dir / f"viz_{i:02d}.png", dpi=150)
        print(f"\nSaved {len(plt.get_fignums())} figure(s) to {args.save_dir}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
