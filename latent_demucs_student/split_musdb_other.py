#!/usr/bin/env python3
"""Split MUSDB 'other' stems into guitar/piano/other via htdemucs_6s.

For each MUSDB track:
1. Extract 'other.wav' from the zip (contains guitar+piano+keys+strings+etc)
2. Run htdemucs_6s on it
3. Take the guitar and piano channels from demucs output
4. Compute residual_other = original_other - guitar - piano (audio domain)
5. VAE-encode guitar, piano, residual_other
6. Save as guitar.vae.pt, piano.vae.pt, other_split.vae.pt

After this, each MUSDB track has 6 stems:
  drums.vae.pt     (real GT)
  bass.vae.pt      (real GT)
  vocals.vae.pt    (real GT)
  guitar.vae.pt    (htdemucs_6s from 'other')
  piano.vae.pt     (htdemucs_6s from 'other')
  other_split.vae.pt (residual: other - guitar - piano)

Usage:
    python split_musdb_other.py
"""
import os
import sys
import time
import traceback
import zipfile
from pathlib import Path

import torch
import torchaudio
import soundfile as sf
import numpy as np

sys.path.insert(0, "/scratch/ACE-Step-1.5")

MUSDB_ZIP = "/scratch/musdb18hq.zip"
LATENT_ROOT = Path("/scratch/musdb18_latents")
SAMPLE_RATE = 48000


def load_vae():
    from diffusers.models import AutoencoderOobleck
    vae = AutoencoderOobleck.from_pretrained("/scratch/ACE-Step-1.5/checkpoints/vae")
    return vae.cuda().eval().to(torch.bfloat16)


def load_demucs():
    from demucs.pretrained import get_model
    m = get_model("htdemucs_6s")
    m.cuda().eval()
    print(f"[demucs] sources: {m.sources}")
    return m


def vae_encode(vae, audio_48k_stereo):
    """audio: [2, N] float32 @ 48k → [T, 64] float32 cpu."""
    x = audio_48k_stereo.unsqueeze(0).cuda().to(torch.bfloat16)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
    return z.squeeze(0).permute(1, 0).cpu().float()


def run_demucs(model, audio_44k_stereo):
    """audio: [2, N] float32 @ 44.1k → dict {stem: [2, N] @ 44.1k}."""
    from demucs.apply import apply_model
    x = audio_44k_stereo.unsqueeze(0).cuda()
    with torch.no_grad():
        sources = apply_model(model, x, device="cuda", split=True,
                              overlap=0.1, progress=False)[0]
    return {name: sources[i].cpu() for i, name in enumerate(model.sources)}


def main():
    print("[split] Loading VAE...")
    vae = load_vae()
    print("[split] Loading htdemucs_6s...")
    demucs = load_demucs()

    print("[split] Opening MUSDB zip...")
    zf = zipfile.ZipFile(MUSDB_ZIP)

    # Find all tracks
    tracks = set()
    for name in zf.namelist():
        if name.endswith("/other.wav"):
            # e.g. "train/Skelpolu - Together Alone/other.wav"
            track_rel = os.path.dirname(name)  # "train/Skelpolu - Together Alone"
            tracks.add(track_rel)
    tracks = sorted(tracks)
    print(f"[split] {len(tracks)} tracks with other.wav")

    # Filter to tracks that don't already have guitar.vae.pt
    todo = []
    for track_rel in tracks:
        lat_dir = LATENT_ROOT / track_rel
        if (lat_dir / "guitar.vae.pt").exists():
            continue
        todo.append(track_rel)
    print(f"[split] {len(todo)} tracks to process ({len(tracks) - len(todo)} already done)")

    ok, fail = 0, 0
    t0 = time.time()
    for i, track_rel in enumerate(todo):
        lat_dir = LATENT_ROOT / track_rel
        lat_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Extract other.wav from zip
            other_wav_path = f"{track_rel}/other.wav"
            with zf.open(other_wav_path) as f:
                audio_np, sr = sf.read(f, dtype="float32")
            # audio_np: [N, C]
            audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None]).float()
            if audio.shape[0] == 1:
                audio = audio.repeat(2, 1)
            elif audio.shape[0] > 2:
                audio = audio[:2]

            # 2. Resample to 44.1k for demucs
            if sr != 44100:
                audio_44 = torchaudio.functional.resample(audio, sr, 44100)
            else:
                audio_44 = audio

            # 3. Run htdemucs_6s on the "other" stem
            stems_44 = run_demucs(demucs, audio_44)
            # htdemucs_6s sources: ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']

            guitar_44 = stems_44["guitar"]
            piano_44 = stems_44["piano"]
            # Residual other = original - guitar - piano (in 44.1k domain)
            other_demucs_44 = stems_44["other"]
            # Also grab any drums/bass/vocals leakage and add to residual
            residual_44 = audio_44.clone()
            residual_44 -= guitar_44
            residual_44 -= piano_44

            # 4. Resample each to 48k and VAE-encode
            for stem_name, wav_44 in [("guitar", guitar_44),
                                       ("piano", piano_44),
                                       ("other_split", residual_44)]:
                wav_48 = torchaudio.functional.resample(wav_44, 44100, SAMPLE_RATE)
                z = vae_encode(vae, wav_48)
                torch.save({"latents": z, "stem": stem_name},
                           lat_dir / f"{stem_name}.vae.pt")

            ok += 1
            elapsed = time.time() - t0
            rate = (ok + fail) / max(elapsed, 1)
            print(f"[{i+1}/{len(todo)}] {track_rel}  ok  "
                  f"(ok={ok} fail={fail} {rate:.2f}/s)", flush=True)

        except Exception as e:
            fail += 1
            print(f"[{i+1}/{len(todo)}] {track_rel}  ERROR: {e}", flush=True)
            traceback.print_exc()

    zf.close()
    print(f"\n[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
