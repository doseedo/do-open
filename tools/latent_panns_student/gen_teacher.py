#!/usr/bin/env python3
"""Distil PANNs from latents alone: pull only *.vae.pt files from the bucket,
decode locally with the Oobleck VAE, then run PANNs CNN14 on the decoded audio.

Writes /scratch/latent_panns_student/cache/{sha1}.pt with
    latent  : [750, 64] fp16    — 30 s window (for the student)
    scores  : [527] fp16        — PANNs clipwise soft target
    ...meta
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as TAF

PAIRS  = "/scratch/latent_panns_student/pairs.json"
CACHE  = Path("/scratch/latent_panns_student/cache")
INDEX  = "/scratch/latent_panns_student/cache_index.json"

VAE_SR = 48000                            # Oobleck decoder output
VAE_SAMPLES_PER_FRAME = 1920              # 48000 / 25
PANNS_SR = 32000
PANNS_CHUNK_S = 10
PANNS_CHUNK_N = PANNS_SR * PANNS_CHUNK_S  # 320000
LATENT_FPS = 25
LATENT_CHUNK_FRAMES = LATENT_FPS * 30     # 750 (student input window)

CACHE.mkdir(parents=True, exist_ok=True)


def pair_id(audio: str) -> str:
    return hashlib.sha1(audio.encode("utf-8")).hexdigest()[:20]


def load_latent(path: str) -> torch.Tensor:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64 and z.shape[1] != 64:
        z = z.t()
    return z.float().contiguous()          # [T, 64] fp32


def load_vae():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae").cuda().eval().to(torch.bfloat16)
    for p in vae.parameters():
        p.requires_grad = False
    return vae


@torch.no_grad()
def decode_to_panns_wav(vae, lat: torch.Tensor) -> torch.Tensor:
    """Decode a crop of the latent (≤10 s audio worth) and return [N] fp32
    mono @ 32 kHz, padded / cropped to PANNS_CHUNK_N samples."""
    # 10 s of 25 Hz latent = 250 frames
    n_lat = min(lat.shape[0], 250)
    z = lat[:n_lat].transpose(0, 1).unsqueeze(0)              # [1, 64, n_lat]
    z = z.to("cuda", dtype=torch.bfloat16)
    audio = vae.decode(z).sample                              # [1, 2, N@48k]
    audio = audio.squeeze(0).float()                          # [2, N]
    mono = audio.mean(dim=0)                                  # [N]
    wav32 = TAF.resample(mono.cpu(), VAE_SR, PANNS_SR)        # [N@32k]
    if wav32.shape[0] < PANNS_CHUNK_N:
        wav32 = F.pad(wav32, (0, PANNS_CHUNK_N - wav32.shape[0]))
    else:
        wav32 = wav32[:PANNS_CHUNK_N]
    return wav32.contiguous()


def crop_latent_window(lat: torch.Tensor, target: int, rng) -> torch.Tensor:
    T = lat.shape[0]
    if T <= target:
        return F.pad(lat, (0, 0, 0, target - T))
    start = int(rng.integers(0, T - target + 1))
    return lat[start:start + target]


def process_batch(items, vae, at):
    wavs = np.stack([it["wav32"].numpy() for it in items])    # [B, 320000]
    clipwise, _ = at.inference(wavs)                          # [B, 527]
    for it, scores in zip(items, clipwise):
        torch.save({
            "latent": it["latent"].half(),                    # [750, 64]
            "scores": torch.from_numpy(scores).half(),        # [527]
            "audio":  it["audio"],
            "latent_path": it["latent_path"],
            "group":  it["group"],
            "duration_s": it["duration_s"],
        }, it["out"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs",  default=PAIRS)
    ap.add_argument("--batch",  type=int, default=32)
    ap.add_argument("--limit",  type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from panns_inference import AudioTagging
    print("[panns-gen-vae] loading VAE + PANNs…", flush=True)
    vae = load_vae()
    at  = AudioTagging(checkpoint_path=None, device="cuda")
    print("[panns-gen-vae] ready", flush=True)

    pairs = json.load(open(args.pairs))
    if args.limit:
        pairs = pairs[:args.limit]

    rng = np.random.default_rng(0)
    t0 = time.time()
    ok = fail = 0
    batch = []

    def flush():
        nonlocal batch, ok, fail
        if not batch:
            return
        try:
            process_batch(batch, vae, at)
            ok += len(batch)
        except Exception as e:
            print(f"  batch err: {e}", flush=True)
            fail += len(batch)
        batch = []

    for i, p in enumerate(pairs):
        out_path = CACHE / f"{pair_id(p['audio'])}.pt"
        if out_path.exists() and not args.overwrite:
            ok += 1
            continue
        try:
            lat = load_latent(p["latent"])                    # [T, 64]
            if lat.shape[0] < 50:
                fail += 1
                continue
            lat_win = crop_latent_window(lat, LATENT_CHUNK_FRAMES, rng)  # [750, 64]
            wav32   = decode_to_panns_wav(vae, lat_win)                  # [320000]
            batch.append({
                "latent": lat_win,
                "wav32":  wav32,
                "audio":  p["audio"],
                "latent_path": p["latent"],
                "group":  p.get("group", ""),
                "duration_s": float(lat.shape[0]) / LATENT_FPS,
                "out":    out_path,
            })
            if len(batch) >= args.batch:
                flush()
        except Exception as e:
            fail += 1
            if fail <= 10:
                print(f"  {p['audio']}: {e}", flush=True)

        if (i + 1) % 200 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            print(f"[{i+1}/{len(pairs)}] ok~{ok} fail={fail} "
                  f"{rate:.1f}/s elapsed={el:.0f}s", flush=True)

    flush()
    idx = []
    for p in pairs:
        out = CACHE / f"{pair_id(p['audio'])}.pt"
        if out.exists():
            idx.append({"path": str(out),
                        "audio": p["audio"],
                        "group": p.get("group", "")})
    json.dump(idx, open(INDEX, "w"))
    print(f"[done] cached={len(idx)} fail={fail} "
          f"elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
