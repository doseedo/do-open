#!/usr/bin/env python3
"""Extract mixture.wav from musdb18hq.zip for every track that already has
latents at /scratch/musdb18_latents/<split>/<track>/mixture.vae.pt.
Saves to /scratch/musdb18_wavs/<split>/<track>/mixture.wav.
"""
import os, io, zipfile, time
from pathlib import Path

ZIP = "/scratch/musdb18hq.zip"
LAT = Path("/scratch/musdb18_latents")
OUT = Path("/scratch/musdb18_wavs")

tracks = []
for mix in LAT.rglob("mixture.vae.pt"):
    rel = mix.parent.relative_to(LAT)            # split/track
    tracks.append(rel)
print(f"[extract] {len(tracks)} tracks have latents")

zf = zipfile.ZipFile(ZIP, "r")
names = set(zf.namelist())

ok, miss, t0 = 0, 0, time.time()
for i, rel in enumerate(tracks):
    out = OUT / rel / "mixture.wav"
    if out.exists():
        ok += 1
        continue
    entry = f"{rel}/mixture.wav"
    if entry not in names:
        # try zip slash flavors
        cand = [n for n in names if n.endswith(f"/{rel}/mixture.wav")
                or n == f"{rel}/mixture.wav"]
        if not cand:
            print(f"  miss: {rel}")
            miss += 1; continue
        entry = cand[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(entry) as src, open(out, "wb") as dst:
        dst.write(src.read())
    ok += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(tracks)} elapsed={time.time()-t0:.0f}s")

print(f"[done] ok={ok} miss={miss} elapsed={time.time()-t0:.0f}s")
print(f"size: {sum(p.stat().st_size for p in OUT.rglob('*.wav'))/1e9:.2f} GB")
