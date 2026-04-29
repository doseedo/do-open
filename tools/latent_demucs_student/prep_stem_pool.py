#!/usr/bin/env python3
"""Pre-decode a pool of random Latents2 stems to wav for mixing.

Decodes ~500 stems per category (guitar/piano/other) through the Oobleck
VAE and saves as 48kHz stereo wavs. These get mixed with MUSDB stem wavs
at training time to create synthetic 6-stem mixtures.

Also extracts individual MUSDB stem wavs (drums/bass/vocals) from the zip.

Output:
    /scratch/latent_demucs_student/stem_pool/
        guitar/0000.wav ... guitar/0499.wav
        piano/0000.wav  ... piano/0499.wav
        other/0000.wav  ... other/0499.wav
        guitar/0000.latent.pt ... (matching latents for targets)
        ...

    /scratch/musdb18_wavs/{train,test}/TrackName/
        drums.wav, bass.wav, vocals.wav   (extracted from zip)

Usage:
    python prep_stem_pool.py [--n-per-cat 500]
"""
import argparse
import os
import random
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")

STEM_INDEX = "/scratch/latent_demucs_student/stem_group_index.pt"
MUSDB_ZIP = "/scratch/musdb18hq.zip"
POOL_DIR = Path("/scratch/latent_demucs_student/stem_pool")
MUSDB_WAV_DIR = Path("/scratch/musdb18_wavs")
LATENT_KEYS = ("latents", "latent", "mix_lat")

# Frozen quality-filter models
VISUAL_CKPT = "/scratch/latent_visual_ckpts/latent_visual_final.pt"
VISUAL_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_visual")
SEM_CKPT = "/scratch/latent_semantic_encoder/ckpts/semantic_final.pt"
SEM_DIR = os.path.join(os.path.dirname(__file__), "..", "latent_semantic_encoder")


def load_vae():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
    return vae.cuda().eval().to(torch.bfloat16)


def load_rms_model():
    """Load frozen RMS/visual model for silence detection."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "latent_visual_infer", os.path.join(VISUAL_DIR, "infer.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(VISUAL_CKPT, map_location="cpu", weights_only=False)
    m = mod.LatentToPeakEnvelope().cuda().eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    return m


def load_sem_model():
    """Load frozen semantic encoder for instrument verification."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sem_model", os.path.join(SEM_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sd = torch.load(SEM_CKPT, map_location="cpu", weights_only=False)
    m = mod.SemanticEncoderWithHeads().cuda().eval()
    m.load_state_dict(sd["model"])
    for p in m.parameters():
        p.requires_grad = False
    return m


def is_good_stem(z, rms_model, sem_model, expected_cat, min_rms=0.02):
    """Filter out silent or misclassified stems.

    Returns True if the stem is loud enough and the semantic encoder
    doesn't flag it as something unexpected.
    """
    x = z.t().unsqueeze(0).cuda()  # [1, 64, T]
    with torch.no_grad():
        # RMS check: predict envelope, reject if mostly silent
        rms_pred = rms_model(x)  # [1, T, 2] or [1, 2, T]
        if rms_pred.dim() == 3 and rms_pred.shape[-1] == 2:
            rms_vals = rms_pred[0, :, 1]  # max envelope
        else:
            rms_vals = rms_pred[0].mean(dim=0) if rms_pred.shape[1] == 2 else rms_pred[0, :, 0]
        mean_rms = rms_vals.abs().mean().item()
        if mean_rms < min_rms:
            return False, "silent"

        # Semantic check: verify it's not flagged as vocals/drums for guitar/piano/other
        sem_out = sem_model(z.unsqueeze(0).cuda())
        vocal_prob = torch.sigmoid(sem_out["vocal"]).item()
        if expected_cat in ("guitar", "piano", "other") and vocal_prob > 0.8:
            return False, "vocal"

    return True, "ok"


def load_latent(path):
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        for k in LATENT_KEYS:
            if k in raw:
                z = raw[k]
                break
        else:
            return None
    else:
        z = raw
    if z.dim() == 2 and z.shape[0] == 64:
        z = z.t()
    return z.float()  # [T, 64]


def decode_latent_to_wav(vae, z):
    """z: [T, 64] → [2, N] float32 @ 48kHz."""
    x = z.t().unsqueeze(0).cuda().to(torch.bfloat16)  # [1, 64, T]
    with torch.no_grad():
        audio = vae.decode(x).sample.float().cpu().squeeze(0)  # [2, N]
    return audio


def resolve_path(candidates):
    if isinstance(candidates, str):
        return candidates if os.path.exists(candidates) else None
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-cat", type=int, default=500)
    ap.add_argument("--max-secs", type=float, default=30.0,
                    help="max seconds per decoded stem")
    args = ap.parse_args()

    max_frames = int(args.max_secs * 25)

    # ── Phase 1: Decode Latents2 stems to wav pool ────────────────────
    print("[pool] Loading stem index...")
    idx_data = torch.load(STEM_INDEX, map_location="cpu", weights_only=False)
    by_6stem = idx_data["by_6stem"]

    print("[pool] Loading VAE...")
    vae = load_vae()
    print("[pool] Loading RMS model (silence filter)...")
    rms_model = load_rms_model()
    print("[pool] Loading semantic model (content filter)...")
    sem_model = load_sem_model()

    rng = random.Random(42)
    categories = ["guitar", "piano", "other"]

    for cat in categories:
        cat_dir = POOL_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        existing = len(list(cat_dir.glob("*.wav")))
        if existing >= args.n_per_cat:
            print(f"[pool] {cat}: {existing} already exist, skipping")
            continue

        candidates = by_6stem.get(cat, [])
        if not candidates:
            print(f"[pool] {cat}: no stems in index, skipping")
            continue

        rng.shuffle(candidates)
        ok, fail, filtered = 0, 0, 0
        t0 = time.time()

        for cand in candidates:
            if ok >= args.n_per_cat:
                break
            path = resolve_path(cand)
            if path is None:
                fail += 1
                continue
            try:
                z = load_latent(path)
                if z is None or z.shape[0] < 25:  # min 1 sec
                    fail += 1
                    continue
                # Trim to max length
                if z.shape[0] > max_frames:
                    start = rng.randint(0, z.shape[0] - max_frames)
                    z = z[start:start + max_frames]

                # Quality filter: reject silent or misclassified stems
                good, reason = is_good_stem(z, rms_model, sem_model, cat)
                if not good:
                    filtered += 1
                    continue

                audio = decode_latent_to_wav(vae, z)  # [2, N]
                idx_str = f"{ok:04d}"
                sf.write(str(cat_dir / f"{idx_str}.wav"),
                         audio.T.numpy(), 48000)
                # Also save the latent for target use
                torch.save({"latents": z}, cat_dir / f"{idx_str}.latent.pt")
                ok += 1

                if ok % 50 == 0:
                    elapsed = time.time() - t0
                    print(f"  {cat}: {ok}/{args.n_per_cat} decoded "
                          f"({elapsed:.0f}s, {fail} no_file, "
                          f"{filtered} filtered)", flush=True)
            except Exception as e:
                fail += 1
                continue

        print(f"[pool] {cat}: {ok} stems decoded, {fail} no_file, "
              f"{filtered} filtered ({time.time()-t0:.0f}s)")

    # ── Phase 2: Extract MUSDB stem wavs from zip ─────────────────────
    print("\n[musdb] Extracting individual stem wavs from zip...")
    zf = zipfile.ZipFile(MUSDB_ZIP)
    extracted = 0
    for name in zf.namelist():
        # Extract drums.wav, bass.wav, vocals.wav (skip other.wav and mixture.wav)
        basename = os.path.basename(name)
        if basename not in ("drums.wav", "bass.wav", "vocals.wav"):
            continue
        # e.g. "train/TrackName/drums.wav"
        out_path = MUSDB_WAV_DIR / name
        if out_path.exists():
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted += 1

    zf.close()
    n_tracks = len(list(MUSDB_WAV_DIR.rglob("drums.wav")))
    print(f"[musdb] Extracted {extracted} new wavs, "
          f"{n_tracks} tracks have drums.wav")

    print("\n[done]")
    print(f"  Stem pool: {POOL_DIR}")
    for cat in categories:
        n = len(list((POOL_DIR / cat).glob("*.wav")))
        print(f"    {cat}: {n} wavs")
    print(f"  MUSDB wavs: {MUSDB_WAV_DIR}")


if __name__ == "__main__":
    main()
