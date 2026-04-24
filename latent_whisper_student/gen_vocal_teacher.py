#!/usr/bin/env python3
"""Generate per-chunk (latent ↔ whisper-hidden) training files for the vocal
latent-whisper student.

Input:  /scratch/latent_whisper_student/voice_pairs.json
        (list of {audio, latent, vc?}, built by scripts/build_voice_pairs.py
        from master_manifest_v2.3.json)

Output: /scratch/latent_whisper_student/chunks/{session_id}_c{chunk_idx}.pt
        dict with:
            mix_lat       : [64, 750] fp32          (student input)
            encoder_hidden: [1500, d_model] fp16    (hidden-state regression target)
            tokens        : [T_tok] int64           (teacher whisper tokens, incl sot/eot)
            text          : str                     (decoded text for this chunk)
            audio_path    : str
            latent_path   : str
            chunk_idx     : int
            model_name    : str
            d_model       : int

Chunks that produce empty / single-token output are *skipped* by default
(`--keep-empty` disables this) — lots of vocal files have silent tails.

Runs Whisper in fp32 (LayerNorm override conflicts with half casting) but
does the encoder forward inside no_grad — throughput on an A100 with base is
roughly 8–12 chunks/sec.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch


# — whisper / latent spec ---------------------------------------------------
WHISPER_SR = 16000
WHISPER_CHUNK_SEC = 30
WHISPER_CHUNK_SAMPLES = WHISPER_SR * WHISPER_CHUNK_SEC   # 480000
LATENT_FPS = 25
CHUNK_LATENT_FRAMES = LATENT_FPS * WHISPER_CHUNK_SEC     # 750
LATENT_DIM = 64


def session_id(audio_path: str) -> str:
    h = hashlib.sha1(audio_path.encode("utf-8")).hexdigest()[:16]
    name = Path(audio_path).stem.replace(" ", "_")[:48]
    return f"{name}_{h}"


def load_whisper(name: str, device: str):
    import whisper
    m = whisper.load_model(name, device=device,
                           download_root="/scratch/cache/whisper")
    m.eval()
    return m


def load_latent(path: str) -> torch.Tensor:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    z = raw["latents"] if isinstance(raw, dict) else raw
    if z.dim() == 2 and z.shape[0] == 64 and z.shape[1] != 64:
        z = z.t()
    return z.float()                                     # [T, 64]


def load_audio_mono_16k(path: str) -> torch.Tensor:
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", path,
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(WHISPER_SR),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[:200]}")
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(arr)                         # [N]


@torch.no_grad()
def encode_chunk(model, audio_chunk: torch.Tensor, language: str | None):
    """audio_chunk: fp32 mono [<=480000] → (hidden [1500,D] fp16 cpu,
    tokens int64 cpu, text str)."""
    import whisper
    from whisper.decoding import DecodingOptions

    wav = whisper.pad_or_trim(audio_chunk)
    mel = whisper.log_mel_spectrogram(wav, n_mels=model.dims.n_mels).to(
        next(model.parameters()).device)
    mel = mel.unsqueeze(0)                               # [1, 80, 3000]
    enc = model.encoder(mel)                             # [1, 1500, D]

    opts = DecodingOptions(
        task="transcribe",
        language=language,
        without_timestamps=True,
        fp16=(enc.dtype == torch.float16),
    )
    result = model.decode(enc, opts)
    res = result[0] if isinstance(result, list) else result
    tok = torch.tensor(res.tokens, dtype=torch.long) if res.tokens is not None \
        else torch.empty(0, dtype=torch.long)
    return enc.squeeze(0).half().cpu(), tok, res.text


def process_pair(pair: dict, model, args, out_dir: Path) -> tuple[int, int]:
    """Returns (n_chunks_written, n_chunks_skipped)."""
    sid = session_id(pair["audio"])

    # 1. latent
    lat = load_latent(pair["latent"])                    # [T, 64]
    T_lat = lat.shape[0]
    if T_lat < 50:    # <2 s of latent — skip
        return 0, 0

    # 2. audio
    audio = load_audio_mono_16k(pair["audio"])           # [N]
    total_samples = audio.shape[0]

    # 3. number of non-overlapping 30 s chunks comes from the latent length
    n_chunks = max(1, (T_lat + CHUNK_LATENT_FRAMES - 1) // CHUNK_LATENT_FRAMES)

    written = skipped = 0
    for ci in range(n_chunks):
        out_path = out_dir / f"{sid}_c{ci}.pt"
        if out_path.exists() and not args.overwrite:
            written += 1
            continue

        # latent window
        ls = ci * CHUNK_LATENT_FRAMES
        le = ls + CHUNK_LATENT_FRAMES
        lat_w = lat[ls:le]
        if lat_w.shape[0] < CHUNK_LATENT_FRAMES:
            pad = CHUNK_LATENT_FRAMES - lat_w.shape[0]
            lat_w = torch.nn.functional.pad(lat_w, (0, 0, 0, pad))
        mix_lat = lat_w.transpose(0, 1).contiguous()     # [64, 750]

        # audio window
        s = ci * WHISPER_CHUNK_SAMPLES
        e = s + WHISPER_CHUNK_SAMPLES
        if s >= total_samples:
            chunk = torch.zeros(WHISPER_CHUNK_SAMPLES, dtype=torch.float32)
        else:
            chunk = audio[s:e]
            if chunk.shape[0] < WHISPER_CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(
                    chunk, (0, WHISPER_CHUNK_SAMPLES - chunk.shape[0]))

        hidden, tok, text = encode_chunk(model, chunk, args.language)

        # filter silent/tokenless chunks unless --keep-empty
        if not args.keep_empty and len(tok) <= 1:
            skipped += 1
            continue

        torch.save({
            "mix_lat": mix_lat,
            "encoder_hidden": hidden,
            "tokens": tok,
            "text": text,
            "audio_path": pair["audio"],
            "latent_path": pair["latent"],
            "chunk_idx": ci,
            "model_name": args.model,
            "d_model": int(model.dims.n_audio_state),
        }, out_path)
        written += 1

    return written, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="/scratch/latent_whisper_student/voice_pairs.json")
    ap.add_argument("--out",   default="/scratch/latent_whisper_student/chunks")
    ap.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium",
                             "large", "large-v2", "large-v3"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard",  type=int, default=0, help="0-based shard index")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-empty", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = json.load(open(args.pairs))
    if args.nshards > 1:
        pairs = pairs[args.shard::args.nshards]
    if args.limit:
        pairs = pairs[:args.limit]
    print(f"[gen-vocal-teacher] pairs to process: {len(pairs)} "
          f"(shard {args.shard}/{args.nshards})", flush=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[gen-vocal-teacher] loading whisper-{args.model} on {device}…",
          flush=True)
    model = load_whisper(args.model, device=device)
    print(f"[gen-vocal-teacher] d_model={model.dims.n_audio_state} "
          f"layers={model.dims.n_audio_layer}", flush=True)

    ok = fail = total_written = total_skipped = 0
    t0 = time.time()
    for i, p in enumerate(pairs):
        try:
            w, s = process_pair(p, model, args, out_dir)
            total_written += w
            total_skipped += s
            ok += 1
        except Exception as e:
            fail += 1
            if fail <= 10:
                print(f"  ERROR {p['audio']}: {e}", flush=True)

        if (i + 1) % 25 == 0 or i + 1 == len(pairs):
            el = time.time() - t0
            rate = (i + 1) / max(el, 1)
            print(f"[{i+1}/{len(pairs)}] ok={ok} fail={fail} "
                  f"wrote={total_written} skipped={total_skipped} "
                  f"({rate:.2f} pairs/s, {el:.0f}s elapsed)", flush=True)

    print(f"[done] ok={ok} fail={fail} chunks_written={total_written} "
          f"skipped={total_skipped} elapsed={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
