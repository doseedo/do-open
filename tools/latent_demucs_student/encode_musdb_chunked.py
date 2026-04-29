#!/usr/bin/env python3
"""Encode MUSDB18-HQ stems → Oobleck VAE latents, with memory bounding.

Differences from encode_musdb.py:
  - Chunks each stem into CHUNK_SECS-second segments before encoding
    (caps peak GPU memory regardless of song length)
  - Retries on OOM with halved chunk size
  - Skips already-done tracks
  - Frees memory aggressively between encodes
"""
import os, sys, io, gc, zipfile, time
from pathlib import Path

import torch
import torchaudio
import soundfile as sf

ZIP_PATH = "/scratch/musdb18hq.zip"
OUT_ROOT = Path("/scratch/musdb18_latents")
STEMS = ["mixture", "drums", "bass", "vocals", "other"]
SAMPLE_RATE = 48000
CHUNK_SECS = 20    # encode in 20-sec chunks (≈ 1 GB peak)
MIN_CHUNK_SECS = 4

# Cap our process to 6 GB to leave room for the training process
torch.cuda.set_per_process_memory_fraction(0.10, 0)  # ~8 GB on 80 GB A100


def load_vae():
    from diffusers.models import AutoencoderOobleck
    return AutoencoderOobleck.from_pretrained(
        "/scratch/ACE-Step-1.5/checkpoints/vae"
    ).cuda().eval().to(torch.bfloat16)


def encode_chunked(vae, audio_2ch_48k, chunk_secs=CHUNK_SECS):
    """Encode [2, N] @ 48k → [T, 64] by chunking."""
    chunk_samples = chunk_secs * SAMPLE_RATE
    pieces = []
    for s in range(0, audio_2ch_48k.shape[1], chunk_samples):
        piece = audio_2ch_48k[:, s:s + chunk_samples]
        x = piece.unsqueeze(0).cuda().to(torch.bfloat16)
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample()        # [1, 64, T]
        pieces.append(z.squeeze(0).permute(1, 0).cpu().float())  # [T, 64]
        del x, z
        torch.cuda.empty_cache()
    return torch.cat(pieces, dim=0)                       # [T_total, 64]


def encode_with_retry(vae, audio):
    chunk = CHUNK_SECS
    while chunk >= MIN_CHUNK_SECS:
        try:
            return encode_chunked(vae, audio, chunk_secs=chunk)
        except torch.cuda.OutOfMemoryError:
            chunk = chunk // 2
            torch.cuda.empty_cache()
            gc.collect()
            print(f"  OOM, retry with chunk={chunk}s", flush=True)
    raise RuntimeError("encode failed even at min chunk")


def read_wav_from_zip(zf, name):
    with zf.open(name) as f:
        data = f.read()
    audio_np, sr = sf.read(io.BytesIO(data), dtype="float32")
    audio = torch.from_numpy(audio_np.T if audio_np.ndim > 1 else audio_np[None])
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2]
    if sr != SAMPLE_RATE:
        audio = torchaudio.functional.resample(audio, sr, SAMPLE_RATE)
    return audio


def main():
    print("[musdb-c] opening zip...")
    zf = zipfile.ZipFile(ZIP_PATH, "r")
    tracks = {}
    for n in zf.namelist():
        if not n.endswith(".wav"): continue
        parts = n.split("/")
        if len(parts) < 3: continue
        split, name, fname = parts[0], parts[1], parts[-1]
        stem = fname[:-4]
        if stem not in STEMS: continue
        tracks.setdefault(f"{split}/{name}", {})[stem] = n
    tracks = {k: v for k, v in tracks.items() if len(v) == len(STEMS)}
    print(f"[musdb-c] {len(tracks)} complete tracks")

    print("[musdb-c] loading VAE (capped to ~8 GB GPU)...")
    vae = load_vae()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ok, skip, fail, t0 = 0, 0, 0, time.time()

    for i, (key, entries) in enumerate(sorted(tracks.items())):
        out_dir = OUT_ROOT / key
        out_dir.mkdir(parents=True, exist_ok=True)
        if all((out_dir / f"{s}.vae.pt").exists() for s in STEMS):
            skip += 1
            continue
        print(f"[{i+1}/{len(tracks)}] {key}", flush=True)
        try:
            for stem in STEMS:
                target = out_dir / f"{stem}.vae.pt"
                if target.exists(): continue
                wav = read_wav_from_zip(zf, entries[stem])
                z = encode_with_retry(vae, wav)
                torch.save({"latents": z, "stem": stem}, target)
                del wav, z
                gc.collect()
                torch.cuda.empty_cache()
            ok += 1
        except Exception as e:
            print(f"  ERROR {key}: {type(e).__name__}: {str(e)[:200]}", flush=True)
            fail += 1
            torch.cuda.empty_cache()
            gc.collect()

        if (ok + fail) % 5 == 0 and (ok + fail) > 0:
            el = time.time() - t0
            print(f"  ... ok={ok} fail={fail} skip={skip} elapsed={el:.0f}s "
                  f"({(ok+fail)/max(el,1):.2f}/s)", flush=True)

    print(f"[done] ok={ok} skip={skip} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
