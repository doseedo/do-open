#!/usr/bin/env python3
"""
Decode-vs-envelope test — find out whether the "sem4Decoder envelope is way
off from decoded audio" symptom is:
  (a) loading/layout bug in JS (channel swap, scale, WAV-roundtrip mismatch)
  (b) envelope function mismatch with the canonical _envelopeFromAudioBuffer
  (c) sem_decoder model simply producing subpar stems — nothing to fix in JS

What it does (offline, no browser, no backend call):
  1. Load cached ONNX models (distill_demucs, sem_demucs, sem_decoder)
  2. Decode the first 8 s of tears.mp3 through the same pipeline sem4Decoder.js
     uses (DC-remove → sem_demucs embedding → distill_demucs → sem_decoder ×4)
  3. For each stem, produce THREE envelopes that ought to be byte-identical
     if loading is correct:
       A = peak-hold over the raw decoder Float32 output, L/R channels-first
           (= envelopeOfStereoChunk in sem4Decoder.js)
       B = peak-hold over the same samples written to a 16-bit PCM WAV and
           decoded back with scipy.io.wavfile (= WAV roundtrip like browser)
       C = peak-hold computed via independent numpy path on the raw output
           (sanity — should equal A)
     Plus:
       D = peak-hold of the MIX input over the same time window (reference)
  4. Report per-stem correlations, max |A-B| + |A-C|, plus absolute scale of
     each envelope. A->B discrepancy larger than 16-bit quantization noise
     means we've got a loading bug in the JS WAV writer or envelope reader.

Usage:
    python3 scripts/decode_vs_envelope_test.py
"""
from __future__ import annotations

import os
import sys
import io
import wave
import subprocess
from pathlib import Path
import numpy as np
import onnxruntime as ort

CACHE = Path("/tmp/doseedo_verify_models")
MP3 = os.environ.get(
    "SMOKE_MP3",
    "/Users/hydroadmin/Downloads/tearsforfearseverybodywantstoruletheworldofficia (1).mp3",
)
OUT = Path("/tmp/decode_vs_envelope")
OUT.mkdir(parents=True, exist_ok=True)

SR = 48000
ENV_HOP = 512  # 93.75 fps — matches sem4Decoder + _envelopeFromAudioBuffer


def load_audio_48k_stereo(path: str, seconds: float = 8.0) -> np.ndarray:
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path, "-f", "f32le", "-ac", "2", "-ar", str(SR), "-"
    ])
    x = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T  # [2, N]
    x = x - x.mean(axis=1, keepdims=True)                       # DC-remove
    n = int(seconds * SR)
    n = (n // 1920) * 1920
    return x[:, :n].astype(np.float32)


def env_peak_hold_channels_first(flat_2N: np.ndarray, numSamples: int) -> np.ndarray:
    """EXACT port of sem4Decoder.envelopeOfStereoChunk (post peak-hold fix).
    flat_2N: shape (2*N,), layout [L0..L_{N-1}, R0..R_{N-1}]
    returns (2*T,) with first T = -peak, last T = +peak.
    """
    N = numSamples
    T = N // ENV_HOP
    env = np.zeros(2 * T, dtype=np.float32)
    for t in range(T):
        off = t * ENV_HOP
        L = flat_2N[off : off + ENV_HOP]
        R = flat_2N[N + off : N + off + ENV_HOP]
        peak = float(max(np.abs(L).max(initial=0.0), np.abs(R).max(initial=0.0)))
        env[t]     = -peak
        env[T + t] = peak
    return env


def env_peak_hold_from_wav_bytes(wav_bytes: bytes) -> np.ndarray:
    """EXACT port of DAWOptimized._envelopeFromAudioBuffer.
    Decode 16-bit PCM stereo WAV → per-channel float arrays, peak-hold over
    both channels into ±peak at 93.75 fps.
    """
    bio = io.BytesIO(wav_bytes)
    with wave.open(bio, "rb") as w:
        nCh = w.getnchannels(); sw = w.getsampwidth(); sr = w.getframerate()
        nFrames = w.getnframes()
        raw = w.readframes(nFrames)
    assert sw == 2, f"expected 16-bit PCM, got {sw*8}-bit"
    assert sr == SR, f"expected {SR} Hz, got {sr} Hz"
    ints = np.frombuffer(raw, dtype="<i2").reshape(-1, nCh)
    chans = [ints[:, c].astype(np.float32) / 32767.0 for c in range(nCh)]
    N = chans[0].size
    hop = max(1, int(round(sr / (SR / ENV_HOP))))  # = 512 for sr=48k
    T = N // hop
    env = np.zeros(2 * T, dtype=np.float32)
    for t in range(T):
        s, e = t * hop, t * hop + hop
        peak = 0.0
        for c in range(nCh):
            a = np.abs(chans[c][s:e])
            if a.size:
                m = float(a.max())
                if m > peak:
                    peak = m
        env[t]     = -peak
        env[T + t] = peak
    return env


def stereo_flat_to_wav_bytes(flat_2N: np.ndarray, numSamples: int) -> bytes:
    """EXACT port of stereoBufferToWavUrl — 16-bit PCM stereo interleaved."""
    N = numSamples
    L = np.clip(flat_2N[0:N], -1.0, 1.0)
    R = np.clip(flat_2N[N:2 * N], -1.0, 1.0)
    # int16 with JS's "(l*0x7fff)|0" truncation
    Li = np.clip((L * 0x7fff).astype(np.int32), -0x8000, 0x7fff).astype(np.int16)
    Ri = np.clip((R * 0x7fff).astype(np.int32), -0x8000, 0x7fff).astype(np.int16)
    interleaved = np.empty(N * 2, dtype=np.int16)
    interleaved[0::2] = Li
    interleaved[1::2] = Ri
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(interleaved.tobytes())
    return bio.getvalue()


def summarize(env: np.ndarray, name: str) -> None:
    T = env.size // 2
    pk = env[T:]
    print(f"  {name:22s}  T={T}  peak(max)={pk.max():.4f}  peak(mean)={pk.mean():.4f}")


def main() -> int:
    mix = load_audio_48k_stereo(MP3, seconds=8.0)
    N = mix.shape[1]
    T = N // 1920
    print(f"input: {mix.shape} ({N/SR:.2f}s, T={T} latent frames)")

    so = ort.SessionOptions(); so.log_severity_level = 3
    kws = dict(sess_options=so, providers=["CPUExecutionProvider"])
    sem = ort.InferenceSession(str(CACHE / "sem_demucs_packed.onnx"), **kws)
    dem = ort.InferenceSession(str(CACHE / "distill_demucs_fp16.onnx"), **kws)
    dec = ort.InferenceSession(str(CACHE / "sem_decoder_fp16.onnx"), **kws)

    audio = mix[None]  # [1,2,N]
    emb = sem.run(None, {"waveform": audio})[
        [o.name for o in sem.get_outputs()].index("embedding")
    ]
    stem_latents = dem.run(None, {"audio": audio})[0]
    assert emb.shape == (1, 4, 128)
    assert stem_latents.shape == (1, 4, 64, T)

    # mix reference envelope
    mix_flat = np.concatenate([mix[0], mix[1]]).astype(np.float32)  # channels-first
    env_mix = env_peak_hold_channels_first(mix_flat, N)

    names = ["drums", "bass", "vocals", "other"]
    max_loading_err = 0.0
    for s, name in enumerate(names):
        lat = stem_latents[:, s].astype(np.float32)
        e = emb[:, s].astype(np.float32)
        out = dec.run(None, {"latent": lat, "sem_emb": e})[0]
        # [1, 2, 1920*T] — onnx row-major → contiguous channels-first
        assert out.shape == (1, 2, 1920 * T), f"{name} shape {out.shape}"
        flat = out.reshape(-1).astype(np.float32)  # [2*N] channels-first
        assert flat.size == 2 * N

        env_A = env_peak_hold_channels_first(flat, N)
        wav = stereo_flat_to_wav_bytes(flat, N)
        env_B = env_peak_hold_from_wav_bytes(wav)
        # numpy-vectorised sanity check
        L = flat[:N].reshape(N // ENV_HOP, ENV_HOP)
        R = flat[N:].reshape(N // ENV_HOP, ENV_HOP)
        per_frame_peak = np.maximum(np.abs(L).max(axis=1), np.abs(R).max(axis=1))
        env_C = np.concatenate([-per_frame_peak, per_frame_peak]).astype(np.float32)

        # ── save wav for manual listen ───────────────────────────────────
        (OUT / f"{name}.wav").write_bytes(wav)

        # ── error metrics ────────────────────────────────────────────────
        T_env = env_A.size // 2
        ab = float(np.max(np.abs(env_A - env_B))) if env_A.shape == env_B.shape else float("inf")
        ac = float(np.max(np.abs(env_A - env_C)))
        # 16-bit WAV roundtrip quantizes at ~1/32768 ≈ 3e-5 — expect ab < ~5e-5
        quant = 1.0 / 32767.0
        ok_ab = ab < 5 * quant
        ok_ac = ac < 1e-6
        print()
        print(f"[{name}]")
        summarize(env_A, "A (raw f32, channels-first)")
        summarize(env_B, "B (wav roundtrip, 16-bit)  ")
        summarize(env_C, "C (numpy-vec sanity)       ")
        print(f"  max|A-B| = {ab:.2e}  {'OK (16-bit quant)' if ok_ab else 'FAIL'}")
        print(f"  max|A-C| = {ac:.2e}  {'OK' if ok_ac else 'FAIL (scalar vs vector mismatch)'}")
        if not ok_ab:
            max_loading_err = max(max_loading_err, ab)
        # correlation vs input mix envelope (just a data point for user)
        T_env_mix = env_mix.size // 2
        if T_env == T_env_mix:
            corr = float(np.corrcoef(env_A[T_env:], env_mix[T_env_mix:])[0, 1])
            print(f"  corr(peak_stem, peak_mix) = {corr:+.3f}")

    # save mix too
    mix_wav = stereo_flat_to_wav_bytes(mix_flat, N)
    (OUT / "_mix.wav").write_bytes(mix_wav)

    print()
    print(f"wrote decoded stems + mix to {OUT}/")
    if max_loading_err > 0:
        print(f"⚠ loading path shows discrepancy > 16-bit quant noise (max {max_loading_err:.2e})")
        print("  That would be a real JS loading bug. Investigate envelopeOfStereoChunk/_envelopeFromAudioBuffer.")
        return 1
    print("✓ JS envelope path matches WAV-roundtrip path within 16-bit quantization.")
    print("  → 'envelope off' is not a loading bug; it is the decoder's output quality.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
