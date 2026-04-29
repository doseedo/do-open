#!/usr/bin/env python3
"""Benchmark: latent-lyric student vs. Whisper (audio).

Compares end-to-end transcription wall time on the same clips. Warm-up
runs excluded from the timing.

Usage:
    python bench_vs_whisper.py \
        --ckpt /scratch/latent_whisper_student/ckpts_vocal/student_final.pt \
        --whisper base \
        --n 20
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime import LatentLyricRuntime

WHISPER_SR = 16000


def decode_audio_ffmpeg(path: str, sr: int = WHISPER_SR) -> np.ndarray:
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(sr),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=True)
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def time_student(rt, latent_cpu):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    texts = rt.transcribe(latent_cpu, language="en", max_len=256)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, texts


def time_whisper(model, audio_np, lang="en"):
    # model.transcribe() = full pipeline incl. mel, encoder, decoder, detok
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = model.transcribe(audio_np, language=lang, fp16=False,
                              without_timestamps=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0, result["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # — samples ---------------------------------------------------------
    cache_idx = json.load(open("/scratch/latent_whisper_student/cache_index.json"))
    rng = random.Random(args.seed)
    picks = rng.sample(cache_idx, min(args.n + args.warmup, len(cache_idx)))

    samples = []
    for entry in picks:
        d = torch.load(entry["path"], map_location="cpu", weights_only=False)
        audio_path = d["audio"]
        if not os.path.exists(audio_path):
            continue
        try:
            audio = decode_audio_ffmpeg(audio_path, WHISPER_SR)
        except Exception:
            continue
        samples.append({
            "latent": d["latent"],
            "audio":  audio,
            "dur_s":  float(d.get("duration_s", len(audio) / WHISPER_SR)),
            "audio_path": audio_path,
        })
        if len(samples) >= args.n + args.warmup:
            break
    print(f"[bench] loaded {len(samples)} samples", flush=True)
    if len(samples) <= args.warmup:
        print("not enough samples")
        return

    # — models ----------------------------------------------------------
    print(f"[bench] loading student {args.ckpt}…", flush=True)
    rt = LatentLyricRuntime(args.ckpt)

    print(f"[bench] loading whisper-{args.whisper}…", flush=True)
    import whisper
    w = whisper.load_model(args.whisper, device="cuda",
                           download_root="/scratch/cache/whisper")
    w.eval()

    # — warm-up ---------------------------------------------------------
    for s in samples[:args.warmup]:
        _ = time_student(rt, s["latent"])
        _ = time_whisper(w, s["audio"])

    # — timed runs ------------------------------------------------------
    rows = []
    for i, s in enumerate(samples[args.warmup:]):
        ts, txt_s = time_student(rt, s["latent"])
        tw, txt_w = time_whisper(w, s["audio"])
        rows.append({
            "dur_s":    s["dur_s"],
            "student":  ts,
            "whisper":  tw,
            "speedup":  tw / max(ts, 1e-9),
            "txt_s":    txt_s[0] if isinstance(txt_s, list) else txt_s,
            "txt_w":    txt_w,
        })
        print(f"[{i+1}/{len(samples)-args.warmup}]  "
              f"dur={s['dur_s']:6.1f}s  "
              f"student={ts*1000:7.1f}ms  "
              f"whisper={tw*1000:7.1f}ms  "
              f"speedup={tw/max(ts,1e-9):5.1f}×",
              flush=True)

    # — summary ---------------------------------------------------------
    st = np.array([r["student"] for r in rows])
    wh = np.array([r["whisper"] for r in rows])
    du = np.array([r["dur_s"]   for r in rows])

    print()
    print("============ SUMMARY ============")
    print(f"n samples       : {len(rows)}")
    print(f"total audio     : {du.sum():.1f}s")
    print(f"student total   : {st.sum()*1000:.1f}ms   "
          f"mean={st.mean()*1000:.1f}ms  p50={np.percentile(st,50)*1000:.1f}ms")
    print(f"whisper total   : {wh.sum()*1000:.1f}ms   "
          f"mean={wh.mean()*1000:.1f}ms  p50={np.percentile(wh,50)*1000:.1f}ms")
    print(f"student RTF     : {(st / du).mean():.4f}  "
          f"(lower = faster than realtime)")
    print(f"whisper RTF     : {(wh / du).mean():.4f}")
    print(f"speedup (mean)  : {(wh / st).mean():.1f}×")
    print(f"speedup (median): {np.median(wh / st):.1f}×")
    print()
    print("============ SAMPLE OUTPUT ============")
    for i, r in enumerate(rows[:3]):
        print(f"--- clip {i+1} ({r['dur_s']:.1f}s) ---")
        print(f"student: {r['txt_s'][:160]}")
        print(f"whisper: {r['txt_w'][:160]}")


if __name__ == "__main__":
    main()
