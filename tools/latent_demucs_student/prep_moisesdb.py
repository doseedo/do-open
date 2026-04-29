#!/usr/bin/env python3
"""Download MoisesDB, encode stems to Oobleck VAE latents.

MoisesDB (240 tracks, 6+ stem categories) replaces MUSDB18-HQ as the
secondary supervised dataset for 6-stem distillation.

Stem mapping to htdemucs_6s order:
  drums, bass, other, vocals, guitar, piano

MoisesDB sub-stems are merged into these 6 categories.

Output structure:
    /scratch/moisesdb_latents/<track_id>/
        mixture.vae.pt          [T, 64]
        drums.vae.pt            [T, 64]
        bass.vae.pt             [T, 64]
        other.vae.pt            [T, 64]
        vocals.vae.pt           [T, 64]
        guitar.vae.pt           [T, 64]
        piano.vae.pt            [T, 64]

    /scratch/moisesdb_wavs/<track_id>/
        mixture.wav             stereo 48kHz (for waveform input to student)

Usage:
    pip install git+https://github.com/moises-ai/moises-db.git
    # Download dataset from https://music.ai/research/ → set MOISESDB_PATH
    python prep_moisesdb.py [--data-path /scratch/moisesdb] [--limit N]
"""
import argparse
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import torchaudio
import soundfile as sf

sys.path.insert(0, "/scratch/ACE-Step-1.5")

LATENT_OUT = Path("/scratch/moisesdb_latents")
WAV_OUT = Path("/scratch/moisesdb_wavs")
STEMS_6 = ["drums", "bass", "other", "vocals", "guitar", "piano"]
SAMPLE_RATE = 48000

# MoisesDB stem taxonomy → 6-stem categories
# MoisesDB has hierarchical stems; we map top-level to our 6 categories.
MOISES_STEM_MAP = {
    "drums": "drums",
    "percussion": "drums",
    "bass": "bass",
    "vocals": "vocals",
    "guitar": "guitar",
    "piano": "piano",
    "bowed_strings": "other",
    "wind": "other",
    "other": "other",
    "other_plucked": "other",
}


def load_vae():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
    return vae.cuda().eval().to(torch.bfloat16)


def vae_encode(vae, audio_48k_stereo):
    """audio: [2, N] float32 @ 48k → [T, 64] float32 cpu."""
    x = audio_48k_stereo.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()  # [1, 64, T]
    return z.squeeze(0).permute(1, 0).cpu().float()  # [T, 64]


def ensure_stereo_48k(audio, sr):
    """Convert audio tensor to stereo 48kHz. audio: [C, N]."""
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio


def process_track(db, track_idx, vae):
    """Process one MoisesDB track: mix stems into 6 categories, encode."""
    track = db[track_idx]
    track_id = track.id if hasattr(track, "id") else f"track_{track_idx:04d}"
    lat_dir = LATENT_OUT / track_id
    wav_dir = WAV_OUT / track_id

    # Skip if already done
    if (lat_dir / "mixture.vae.pt").exists():
        return True, "exists"

    lat_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    # Load stems and group into 6 categories
    sr = db.sample_rate if hasattr(db, "sample_rate") else 44100
    category_audio = {}  # category → [2, N] accumulated audio

    # Get the stem data from MoisesDB
    stems = track.stems  # dict of {stem_name: np.ndarray}
    for stem_name, audio_np in stems.items():
        cat = MOISES_STEM_MAP.get(stem_name)
        if cat is None:
            continue

        # audio_np: [N, C] or [N] numpy
        audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None]).float()
        audio = ensure_stereo_48k(audio, sr)

        if cat in category_audio:
            # Sum into existing (pad/trim to match length)
            existing = category_audio[cat]
            N = max(existing.shape[1], audio.shape[1])
            if existing.shape[1] < N:
                existing = torch.nn.functional.pad(existing, (0, N - existing.shape[1]))
            if audio.shape[1] < N:
                audio = torch.nn.functional.pad(audio, (0, N - audio.shape[1]))
            category_audio[cat] = existing + audio
        else:
            category_audio[cat] = audio

    if not category_audio:
        return False, "no stems"

    # Build the full mixture by summing all category audio
    max_len = max(a.shape[1] for a in category_audio.values())
    mixture = torch.zeros(2, max_len)
    for audio in category_audio.values():
        mixture[:, :audio.shape[1]] += audio

    # Save mixture wav
    sf.write(str(wav_dir / "mixture.wav"),
             mixture.T.numpy(), SAMPLE_RATE)

    # Encode mixture
    z_mix = vae_encode(vae, mixture)
    torch.save({"latents": z_mix}, lat_dir / "mixture.vae.pt")

    # Encode each category stem
    cats_present = []
    for cat, audio in category_audio.items():
        z = vae_encode(vae, audio)
        torch.save({"latents": z, "stem": cat}, lat_dir / f"{cat}.vae.pt")
        cats_present.append(cat)

    return True, f"stems={cats_present}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/scratch/moisesdb",
                    help="Path to extracted MoisesDB dataset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Try to import moisesdb
    try:
        from moisesdb.dataset import MoisesDB
    except ImportError:
        print("ERROR: moisesdb not installed. Run:")
        print("  pip install git+https://github.com/moises-ai/moises-db.git")
        sys.exit(1)

    if not os.path.exists(args.data_path):
        print(f"ERROR: MoisesDB data not found at {args.data_path}")
        print("Download from https://music.ai/research/ and extract to that path,")
        print("or set MOISESDB_PATH env var and pass --data-path.")
        sys.exit(1)

    print("[moisesdb] Loading dataset...")
    db = MoisesDB(data_path=args.data_path, sample_rate=44100)
    n_tracks = len(db)
    print(f"[moisesdb] {n_tracks} tracks")

    print("[moisesdb] Loading VAE...")
    vae = load_vae()

    LATENT_OUT.mkdir(parents=True, exist_ok=True)
    WAV_OUT.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit > 0 else n_tracks
    ok, fail = 0, 0
    t0 = time.time()

    for i in range(min(limit, n_tracks)):
        try:
            success, status = process_track(db, i, vae)
        except Exception as e:
            print(f"[{i+1}/{limit}] ERROR: {e}")
            traceback.print_exc()
            fail += 1
            continue

        if success:
            ok += 1
        else:
            fail += 1

        elapsed = time.time() - t0
        rate = (ok + fail) / max(elapsed, 1)
        print(f"[{i+1}/{limit}] {status}  (ok={ok} fail={fail} {rate:.2f}/s)",
              flush=True)

    print(f"\n[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")
    print(f"Latents: {LATENT_OUT}")
    print(f"Wavs: {WAV_OUT}")


if __name__ == "__main__":
    main()
