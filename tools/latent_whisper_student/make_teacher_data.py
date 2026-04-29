#!/usr/bin/env python3
"""Generate Whisper teacher data for the latent-whisper student.

For each session that already has `full_mix.vae.pt` (Oobleck latents at 25 Hz),
we find the matching `full_mix.flac` audio, split it into 30-second chunks
aligned to the latent timeline, run a frozen Whisper model on each chunk, and
save the encoder hidden states + decoded token ids + text alongside the latent.

Output file (one per session, next to full_mix.vae.pt):
    teacher_whisper_{model}.pt  ::  dict with
        "encoder_hidden" : fp16 tensor [N_chunks, 1500, d_model]
        "tokens"         : list[Tensor]  (variable length, int64, incl SOT/EOT)
        "text"           : list[str]
        "model_name"     : str
        "d_model"        : int
        "chunk_latent_frames" : 750
        "chunk_audio_samples_16k" : 480000
        "n_chunks"       : int
        "duration_s"     : float

At training time the dataset crops the latent to
`mix[chunk*750 : (chunk+1)*750]` so each chunk pairs a 30 s audio window
with its Whisper encoder output.

Usage:
    python make_teacher_data.py --latent-root /scratch/stemphonic/data/ossl_latents \
           --audio-root /scratch/stemphonic/data/ossl_audio \
           --model base --limit 100
"""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE_LAT = 48000
LATENT_FPS = 25
SAMPLES_PER_FRAME_48K = SAMPLE_RATE_LAT // LATENT_FPS          # 1920

WHISPER_SR = 16000
WHISPER_CHUNK_SEC = 30
WHISPER_CHUNK_SAMPLES = WHISPER_SR * WHISPER_CHUNK_SEC         # 480000
WHISPER_N_FRAMES = 1500
CHUNK_LATENT_FRAMES = LATENT_FPS * WHISPER_CHUNK_SEC           # 750


def load_whisper(name: str, device: str = "cuda"):
    import whisper
    m = whisper.load_model(name, device=device,
                           download_root="/scratch/cache/whisper")
    m.eval()
    return m


def find_audio(latent_path: Path, audio_roots: list[Path]) -> Path | None:
    """Look for a sibling or mirrored audio file for this latent."""
    stem = latent_path.name.replace(".vae.pt", "")
    # try sibling first
    for ext in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".webm", ".mkv"):
        sib = latent_path.parent / f"{stem}{ext}"
        if sib.exists():
            return sib
    # try parallel tree in each audio_root
    for root in audio_roots:
        for ext in (".flac", ".wav", ".mp3", ".ogg", ".m4a", ".mp4", ".webm", ".mkv"):
            cand = root / f"{stem}{ext}"
            if cand.exists():
                return cand
    # also try parent-relative (full_mix convention)
    if latent_path.name == "full_mix.vae.pt":
        for root in audio_roots:
            rel = latent_path.parent.relative_to(latent_path.parents[
                len(latent_path.parents) - 1])  # unlikely to matter
            for ext in (".flac", ".wav"):
                cand = root / rel / f"full_mix{ext}"
                if cand.exists():
                    return cand
    return None


def load_audio_mono_16k(path: Path) -> torch.Tensor:
    """Decode any container via ffmpeg → mono fp32 @ 16 kHz. Returns [N]."""
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-i", str(path),
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "1", "-ar", str(WHISPER_SR),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {path}: {proc.stderr.decode(errors='ignore')[:200]}")
    arr = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    return torch.from_numpy(arr)                      # [N]


@torch.no_grad()
def encode_chunk(model, audio_chunk_16k_mono: torch.Tensor, language: str | None):
    """Run whisper encoder + greedy decode on one 30 s chunk.

    audio_chunk_16k_mono: [<=480000] fp32 cpu
    returns (encoder_hidden [1500, d_model] fp16 cpu,
             tokens (1d int64 cpu), text str)
    """
    import whisper
    from whisper.decoding import DecodingOptions

    # pad or trim to exactly 30 s then mel-spec
    wav = whisper.pad_or_trim(audio_chunk_16k_mono)
    mel = whisper.log_mel_spectrogram(wav, n_mels=model.dims.n_mels).to(
        next(model.parameters()).device)
    mel = mel.unsqueeze(0)                           # [1, 80, 3000]

    # encoder hidden
    enc = model.encoder(mel)                         # [1, 1500, d_model]

    # greedy decode with the encoder we just computed (re-uses encoder output)
    opts = DecodingOptions(
        task="transcribe",
        language=language,       # None = auto-detect
        without_timestamps=True,
        fp16=(enc.dtype == torch.float16),
    )
    result = model.decode(enc, opts)
    # decode() returns a list (batch size 1)
    res = result[0] if isinstance(result, list) else result
    tok = torch.tensor(res.tokens, dtype=torch.long) if res.tokens is not None \
        else torch.empty(0, dtype=torch.long)
    return enc.squeeze(0).half().cpu(), tok, res.text


def process_session(latent_path: Path, audio_path: Path, model,
                    model_name: str, language: str | None):
    # 1. load full-session latent (to know the timeline)
    raw = torch.load(latent_path, map_location="cpu", weights_only=False)
    lat = raw["latents"] if isinstance(raw, dict) else raw
    if lat.dim() == 2 and lat.shape[0] == 64 and lat.shape[1] != 64:
        lat = lat.t()                                # → [T, 64]
    T_lat = lat.shape[0]
    duration_s = T_lat / LATENT_FPS

    # 2. load audio (mono, 16 kHz)
    audio = load_audio_mono_16k(audio_path)          # [N]
    total_samples = audio.shape[0]

    # 3. align: number of chunks determined by the latent length
    #    (last chunk is zero-padded if needed)
    n_chunks = (T_lat + CHUNK_LATENT_FRAMES - 1) // CHUNK_LATENT_FRAMES
    if n_chunks == 0:
        return None

    enc_hiddens = []
    tokens_list = []
    texts = []
    for i in range(n_chunks):
        s = i * WHISPER_CHUNK_SAMPLES
        e = s + WHISPER_CHUNK_SAMPLES
        if s >= total_samples:
            chunk = torch.zeros(WHISPER_CHUNK_SAMPLES, dtype=torch.float32)
        else:
            chunk = audio[s:e]
            if chunk.shape[0] < WHISPER_CHUNK_SAMPLES:
                chunk = torch.nn.functional.pad(
                    chunk, (0, WHISPER_CHUNK_SAMPLES - chunk.shape[0]))
        enc, tok, text = encode_chunk(model, chunk, language)
        enc_hiddens.append(enc)
        tokens_list.append(tok)
        texts.append(text)

    return {
        "encoder_hidden": torch.stack(enc_hiddens),  # [N, 1500, d_model] fp16
        "tokens": tokens_list,
        "text": texts,
        "model_name": model_name,
        "d_model": int(model.dims.n_audio_state),
        "chunk_latent_frames": CHUNK_LATENT_FRAMES,
        "chunk_audio_samples_16k": WHISPER_CHUNK_SAMPLES,
        "n_chunks": n_chunks,
        "duration_s": duration_s,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-root", required=True,
                    help="dir containing *.vae.pt latent files "
                         "(e.g. /scratch/stemphonic/data/ossl_latents)")
    ap.add_argument("--audio-root", action="append", default=[],
                    help="dir(s) to search for matching audio (repeatable)")
    ap.add_argument("--latent-glob", default="*.vae.pt",
                    help="glob relative to --latent-root (default *.vae.pt)")
    ap.add_argument("--model", default="base",
                    choices=["tiny", "base", "small", "medium",
                             "large", "large-v2", "large-v3"])
    ap.add_argument("--language", default=None,
                    help="force language (e.g. 'en'); default = auto")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    latent_root = Path(args.latent_root)
    audio_roots = [Path(p) for p in (args.audio_root or [latent_root])]

    out_suffix = f"teacher_whisper_{args.model.replace('-', '_')}.pt"

    latents = sorted(latent_root.rglob(args.latent_glob))
    todo = []
    for lp in latents:
        out_path = lp.parent / f"{lp.name.replace('.vae.pt','')}.{out_suffix}"
        if out_path.exists() and not args.overwrite:
            continue
        todo.append((lp, out_path))
        if args.limit and len(todo) >= args.limit:
            break

    print(f"[teacher-whisper] {len(todo)} sessions to process "
          f"(found {len(latents)} latents)")
    if not todo:
        return

    print(f"[teacher-whisper] loading whisper-{args.model}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_whisper(args.model, device=device)
    print(f"[teacher-whisper] d_model={model.dims.n_audio_state} "
          f"n_layers={model.dims.n_audio_layer} device={device}")

    t0 = time.time()
    ok = fail = 0
    for i, (lp, out_path) in enumerate(todo):
        elapsed = time.time() - t0
        rate = (ok + fail) / max(elapsed, 1)
        print(f"[{i+1}/{len(todo)}] {lp.name}  (ok={ok} fail={fail} "
              f"{rate:.2f}/s)", flush=True)

        audio_path = find_audio(lp, audio_roots)
        if audio_path is None:
            print(f"  miss audio for {lp.name}")
            fail += 1
            continue

        try:
            out = process_session(lp, audio_path, model, args.model, args.language)
            if out is None:
                fail += 1
                continue
            torch.save(out, out_path)
            print(f"  → {out_path.name}  "
                  f"({out['n_chunks']} chunks, {out['duration_s']:.1f}s)")
            ok += 1
        except Exception as e:
            print(f"  ERROR {lp.name}: {e}")
            traceback.print_exc()
            fail += 1

    print(f"[done] ok={ok} fail={fail} elapsed={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
