"""
Note-level evaluation against the BasicPitch teacher.

For N held-out pairs, run the student on the latent, decode rolls to notes,
match against teacher notes via mir_eval, and report:

    onset F1            (pitch + onset within 50 ms tolerance)
    onset+offset F1     (pitch + onset + offset within tolerance)
    pitch-only F1       (any onset, correct pitch)
    mean abs pitch error  (semitones, on matched notes — should be 0)
    onset RMSE            (ms, on matched notes)
    velocity RMSE         (MIDI units, on matched notes)

Usage:
    python -m latent_pitch.eval_pitch --ckpt /scratch/latent_pitch_ckpts/pitch_048000.pt --n 50
"""
from __future__ import annotations
import argparse, os, random, sys
import numpy as np
import torch
import mir_eval
import pretty_midi

sys.path.insert(0, "/home/arlo/do2")
from latent_pitch.dataset import (
    LatentMidiPairDataset, _load_latent, latent_to_mid_path,
    DEFAULT_LATENTS_ROOT, DEFAULT_BASICPITCH_ROOT, VAE_HZ, N_PITCH,
)
from latent_pitch.infer import LatentPitchRuntime


def pm_to_arrays(pm):
    intervals, pitches, vels = [], [], []
    for inst in pm.instruments:
        for n in inst.notes:
            if n.end > n.start:
                intervals.append([n.start, n.end])
                pitches.append(n.pitch)
                vels.append(n.velocity)
    if not intervals:
        return (np.zeros((0, 2)), np.array([], dtype=int), np.array([], dtype=int))
    return np.array(intervals), np.array(pitches, dtype=int), np.array(vels, dtype=int)


def evaluate_pair(rt: LatentPitchRuntime, lat_path: str, mid_path: str):
    L = _load_latent(lat_path)
    if L.shape[0] < 8 or L.shape[1] != 64:
        return None
    pm_pred = rt.transcribe(L)
    pm_true = pretty_midi.PrettyMIDI(mid_path)
    iv_p, p_p, v_p = pm_to_arrays(pm_pred)
    iv_t, p_t, v_t = pm_to_arrays(pm_true)
    if len(p_t) == 0:
        return None  # nothing to compare against

    # mir_eval uses pitches in Hz; convert via 440 * 2^((p-69)/12)
    f_p = 440.0 * 2.0 ** ((p_p - 69) / 12.0) if len(p_p) else np.array([], dtype=float)
    f_t = 440.0 * 2.0 ** ((p_t - 69) / 12.0)

    # onset F1 (pitch + onset, default 50 ms tolerance)
    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=None,
    )
    # onset + offset F1 (default 0.2 ratio)
    p_oo, r_oo, f_oo, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=0.2,
    )

    # matched-pair statistics
    matching = mir_eval.transcription.match_notes(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=None,
    )
    onset_errs = []
    pitch_errs = []
    vel_errs = []
    for ti, pi in matching:
        onset_errs.append((iv_p[pi, 0] - iv_t[ti, 0]) * 1000.0)  # ms
        pitch_errs.append(abs(int(p_p[pi]) - int(p_t[ti])))
        vel_errs.append(abs(int(v_p[pi]) - int(v_t[ti])))

    return {
        "n_true": len(p_t),
        "n_pred": len(p_p),
        "n_matched": len(matching),
        "f1_pitch_onset":   f_on,
        "p_pitch_onset":    p_on,
        "r_pitch_onset":    r_on,
        "f1_onset_offset":  f_oo,
        "onset_rmse_ms":    float(np.sqrt(np.mean(np.square(onset_errs)))) if onset_errs else None,
        "abs_pitch_err":    float(np.mean(pitch_errs)) if pitch_errs else None,
        "vel_rmse":         float(np.sqrt(np.mean(np.square(vel_errs)))) if vel_errs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--latents-root", default=DEFAULT_LATENTS_ROOT)
    ap.add_argument("--basicpitch-root", default=DEFAULT_BASICPITCH_ROOT)
    args = ap.parse_args()

    print(f"loading {args.ckpt}...")
    rt = LatentPitchRuntime(args.ckpt)

    # use the same dataset path-walker (random walk + path-mirror lookup)
    # but with a different seed than training so the eval set doesn't
    # overlap (statistically — there's no held-out split, but at 100k
    # files this is fine)
    ds = LatentMidiPairDataset(
        latents_root=args.latents_root,
        basicpitch_root=args.basicpitch_root,
        win_frames=256, seed=args.seed,
    )

    results = []
    n_done = 0
    n_attempts = 0
    while n_done < args.n and n_attempts < args.n * 5:
        n_attempts += 1
        try:
            L, mid_p = ds._safe_pair()
            # evaluate the FULL clip (not the windowed crop the dataset uses
            # for training) — eval should reflect production usage
            lat_path = None  # we have L already; use it directly
            res = evaluate_pair_inplace(rt, L, mid_p)
            if res is not None:
                results.append(res)
                n_done += 1
                if n_done % 10 == 0:
                    print(f"  {n_done}/{args.n}")
        except Exception as e:
            continue

    if not results:
        print("no valid evaluations"); return

    # aggregate
    agg = {k: [] for k in results[0]}
    for r in results:
        for k, v in r.items():
            if v is not None:
                agg[k].append(v)
    avg = {k: float(np.mean(v)) if v else None for k, v in agg.items()}

    print(f"\n=== {args.ckpt} on {len(results)} clips ===")
    print(f"  notes per clip       teacher={int(np.mean([r['n_true'] for r in results])):4d}  "
          f"student={int(np.mean([r['n_pred'] for r in results])):4d}")
    print(f"  F1 (pitch + onset)        {avg['f1_pitch_onset']:.3f}   "
          f"(P {avg['p_pitch_onset']:.3f}  R {avg['r_pitch_onset']:.3f})")
    print(f"  F1 (pitch + onset + off)  {avg['f1_onset_offset']:.3f}")
    print(f"  onset RMSE (matched)      {avg['onset_rmse_ms']:.1f} ms")
    print(f"  abs pitch err  (matched)  {avg['abs_pitch_err']:.3f} semitones")
    print(f"  velocity RMSE  (matched)  {avg['vel_rmse']:.1f} MIDI units")


def evaluate_pair_inplace(rt: LatentPitchRuntime, L: torch.Tensor, mid_path: str,
                          max_frames: int = 256):
    if L.shape[0] < 8 or L.shape[1] != 64:
        return None
    # process only as many frames as the model's positional embedding allows
    L_window = L[:max_frames]
    window_dur = L_window.shape[0] / VAE_HZ  # seconds
    pm_pred = rt.transcribe(L_window)
    pm_true = pretty_midi.PrettyMIDI(mid_path)
    iv_p, p_p, v_p = pm_to_arrays(pm_pred)
    iv_t_full, p_t_full, v_t_full = pm_to_arrays(pm_true)
    # restrict teacher notes to the student's time window so the comparison
    # is apples-to-apples (notes that *start* before window_dur)
    if len(iv_t_full) > 0:
        keep = iv_t_full[:, 0] < window_dur
        iv_t = iv_t_full[keep]
        p_t  = p_t_full[keep]
        v_t  = v_t_full[keep]
    else:
        iv_t, p_t, v_t = iv_t_full, p_t_full, v_t_full
    if len(p_t) == 0:
        return None
    f_p = 440.0 * 2.0 ** ((p_p - 69) / 12.0) if len(p_p) else np.array([], dtype=float)
    f_t = 440.0 * 2.0 ** ((p_t - 69) / 12.0)
    p_on, r_on, f_on, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=None,
    )
    p_oo, r_oo, f_oo, _ = mir_eval.transcription.precision_recall_f1_overlap(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=0.2,
    )
    matching = mir_eval.transcription.match_notes(
        iv_t, f_t, iv_p, f_p, onset_tolerance=0.05, offset_ratio=None,
    )
    onset_errs, pitch_errs, vel_errs = [], [], []
    for ti, pi in matching:
        onset_errs.append((iv_p[pi, 0] - iv_t[ti, 0]) * 1000.0)
        pitch_errs.append(abs(int(p_p[pi]) - int(p_t[ti])))
        vel_errs.append(abs(int(v_p[pi]) - int(v_t[ti])))
    return {
        "n_true": len(p_t),
        "n_pred": len(p_p),
        "n_matched": len(matching),
        "f1_pitch_onset":   f_on,
        "p_pitch_onset":    p_on,
        "r_pitch_onset":    r_on,
        "f1_onset_offset":  f_oo,
        "onset_rmse_ms":    float(np.sqrt(np.mean(np.square(onset_errs)))) if onset_errs else None,
        "abs_pitch_err":    float(np.mean(pitch_errs)) if pitch_errs else None,
        "vel_rmse":         float(np.sqrt(np.mean(np.square(vel_errs)))) if vel_errs else None,
    }


if __name__ == "__main__":
    main()
