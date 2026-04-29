#!/usr/bin/env python3
"""Encode MUSDB18-HQ stems → Oobleck VAE latents.

MUSDB18-HQ zip layout:
  train/<TrackName>/{mixture,drums,bass,vocals,other}.wav
  test/<TrackName>/{mixture,drums,bass,vocals,other}.wav

Output:
  /scratch/musdb18_latents/<split>/<TrackName>/{mixture,drums,bass,vocals,other}.vae.pt

Streams entries from the zip without unpacking everything.
"""
import os, sys, io, zipfile, time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

ZIP_PATH = "/scratch/musdb18hq.zip"
OUT_ROOT = Path("/scratch/musdb18_latents")
STEMS = ["mixture", "drums", "bass", "vocals", "other"]


def load_vae():
    from diffusers.models import AutoencoderOobleck
    return AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae"
    ).cuda().eval().to(torch.bfloat16)


def encode_audio(vae, audio_2ch_48k):
    x = audio_2ch_48k.unsqueeze(0).cuda().to(torch.bfloat16)   # [1, 2, N]
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()                  # [1, 64, T]
    return z.squeeze(0).permute(1, 0).cpu().float()             # [T, 64]


def read_wav_from_zip(zf, name):
    with zf.open(name) as f:
        data = f.read()
    audio_np, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None])
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr != 48000:
        audio = torchaudio.functional.resample(audio, sr, 48000)
    return audio


def main():
    print("[musdb] opening zip...")
    zf = zipfile.ZipFile(ZIP_PATH, "r")

    # Index by track folder
    tracks = {}   # "{split}/{name}" → {stem: zip_entry}
    for n in zf.namelist():
        if not n.endswith(".wav"): continue
        parts = n.split("/")
        if len(parts) < 3: continue
        split, name, fname = parts[0], parts[1], parts[-1]
        stem = fname[:-len(".wav")]
        if stem not in STEMS: continue
        tracks.setdefault(f"{split}/{name}", {})[stem] = n
    tracks = {k: v for k, v in tracks.items() if len(v) == len(STEMS)}
    print(f"[musdb] {len(tracks)} complete tracks")

    print("[musdb] loading VAE...")
    vae = load_vae()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ok, fail, t0 = 0, 0, time.time()

    for i, (key, entries) in enumerate(sorted(tracks.items())):
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        if all((out_dir / f"{s}.vae.pt").exists() for s in STEMS):
            ok += 1
            continue
        print(f"[{i+1}/{len(tracks)}] {key}", flush=True)
        try:
            for stem in STEMS:
                target = out_dir / f"{stem}.vae.pt"
                if target.exists(): continue
                wav = read_wav_from_zip(zf, entries[stem])
                z = encode_audio(vae, wav)
                torch.save({"latents": z, "stem": stem}, target)
                del wav
            ok += 1
        except Exception as e:
            print(f"  ERROR {key}: {e}", flush=True)
            fail += 1
        if (ok + fail) % 5 == 0:
            el = time.time() - t0
            print(f"  ... {ok+fail}/{len(tracks)} elapsed={el:.0f}s "
                  f"({(ok+fail)/max(el,1):.2f}/s)", flush=True)

    print(f"[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
