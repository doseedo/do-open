"""Doseedo audio watermark service — CPU container running AudioSeal.

Two endpoints on the same container:

  POST /embed    — Bearer-gated. multipart/form-data with field "file"
                   (audio bytes) and form field "payload" (16 hex bytes
                   that uniquely identify the generation).
                   Returns: audio/wav body with the watermark embedded.
                   Headers: x-doseedo-seed (hex of payload).

  POST /detect   — Bearer-gated. multipart/form-data with field "file".
                   Returns JSON:
                     { "found": bool,
                       "seed":  "<32-hex>" | null,
                       "confidence": 0.0–1.0,
                       "duration_sec": float,
                       "scanned_at": ISO-8601 }

  GET  /health   — public.

Auth: same VLLM_API_KEY bearer the rest of the stack uses.

Deploy:  modal deploy modal/modal_watermark.py

Watermark backend: AudioSeal (Meta, Apache-2.0). Light enough to run on
CPU — embed/detect on a 4-min stereo file is ~150 ms on a 4-vCPU box.
We keep a single container warm (min=0 with snapshot) — cold start with
the AudioSeal weights baked into the image is ~6-8 s.

Why FastAPI is at MODULE scope:
Same reason as modal_video_scoring.py — Pydantic v2 needs `UploadFile`
in `globalns` to resolve annotations on FastAPI route registration.
"""

import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import modal

# FastAPI imports MUST be at module scope (see header).
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .pip_install(
        "audioseal==0.1.4",
        "soundfile==0.12.1",
        "numpy>=1.24,<2",
        "torch==2.3.1",
        "torchaudio==2.3.1",
        "fastapi==0.115.0",
        "python-multipart==0.0.9",
    )
    # Pre-pull AudioSeal weights so cold start doesn't depend on
    # huggingface.co reachability.
    .run_commands(
        "python -c \"from audioseal import AudioSeal; "
        "AudioSeal.load_generator('audioseal_wm_16bits'); "
        "AudioSeal.load_detector('audioseal_detector_16bits')\""
    )
)

app = modal.App("doseedo-watermark")


# ---------------------------------------------------------------------------
# Per-container state populated at @modal.enter time
# ---------------------------------------------------------------------------

_STATE = {
    "generator": None,
    "detector": None,
    "api_key": None,
    "sample_rate": 16000,  # AudioSeal native rate; we resample I/O around this
}


# ---------------------------------------------------------------------------
# FastAPI surface
# ---------------------------------------------------------------------------

api = FastAPI(title="doseedo-watermark", version="2026.04")
log = logging.getLogger("doseedo.watermark")
logging.basicConfig(level=logging.INFO)


def _require_auth(authorization: str) -> None:
    expected = _STATE.get("api_key")
    if not expected:
        # Auth not configured server-side — refuse rather than fail open.
        raise HTTPException(status_code=503, detail="auth not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(status_code=401, detail="bad bearer")


def _payload_to_bits(payload_hex: str):
    """Convert 32-char hex (16 bytes = 128 bits) to a torch tensor of ±1
    that AudioSeal accepts as the watermark message. We use the first 16
    bits as the AudioSeal message; the full 128-bit hash is stored in the
    Polygon attestation."""
    import torch

    if len(payload_hex) < 4:
        raise HTTPException(status_code=400, detail="payload too short")
    # AudioSeal-16 takes a 16-bit message. Truncate.
    raw = bytes.fromhex(payload_hex)[:2]
    bits = []
    for b in raw:
        for i in range(8):
            bits.append(1 if (b >> (7 - i)) & 1 else 0)
    while len(bits) < 16:
        bits.append(0)
    return torch.tensor([bits[:16]], dtype=torch.int32)


def _bits_to_hex(bits) -> str:
    """Inverse of _payload_to_bits — pack a 16-bit message back to 4 hex
    chars. Caller pads with the rest of the on-chain payload."""
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return f"{val:04x}"


def _load_audio_to_tensor(data: bytes, target_sr: int):
    """Decode arbitrary audio bytes to a mono float32 tensor at target_sr.
    Falls back to ffmpeg via soundfile -> torchaudio if the input format
    isn't WAV. Returns (waveform[1, T], sample_rate)."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        # soundfile can't handle MP3/Opus directly. Punt to torchaudio.
        path = f"/tmp/wm_in_{uuid.uuid4().hex}"
        with open(path, "wb") as fh:
            fh.write(data)
        try:
            tensor, sr = torchaudio.load(path)
            wav = tensor.numpy().T  # -> (T, channels)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    # Mono mix-down for the model. Stereo embed roundtrips the same payload
    # on both channels in the production pipeline; the detector only needs
    # one channel.
    if wav.ndim == 2 and wav.shape[1] > 1:
        wav = wav.mean(axis=1, keepdims=True)
    elif wav.ndim == 1:
        wav = wav.reshape(-1, 1)
    waveform = torch.from_numpy(wav.T.astype("float32"))  # (1, T)

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform, target_sr


def _tensor_to_wav(waveform, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(buf, waveform.squeeze(0).cpu().numpy(), sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


@api.get("/health")
async def health():
    return JSONResponse({
        "ok": _STATE["generator"] is not None and _STATE["detector"] is not None,
        "model": "audioseal_wm_16bits / audioseal_detector_16bits",
        "sample_rate": _STATE["sample_rate"],
    })


@api.post("/embed")
async def embed_endpoint(
    file: UploadFile = File(...),
    payload: str = Form(...),
    authorization: str = Header(default=""),
):
    _require_auth(authorization)
    import torch

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    waveform, sr = _load_audio_to_tensor(data, _STATE["sample_rate"])
    msg_bits = _payload_to_bits(payload)
    with torch.no_grad():
        # AudioSeal generator takes (B, 1, T) and returns watermark to add.
        wm = _STATE["generator"].get_watermark(
            waveform.unsqueeze(0),
            sample_rate=sr,
            message=msg_bits,
        )
        watermarked = waveform.unsqueeze(0) + wm

    body = _tensor_to_wav(watermarked.squeeze(0), sr)
    log.info(
        "embed ok bytes_in=%d bytes_out=%d payload=%s",
        len(data), len(body), payload[:16],
    )
    return Response(
        content=body,
        media_type="audio/wav",
        headers={"x-doseedo-seed": payload, "x-doseedo-sr": str(sr)},
    )


@api.post("/detect")
async def detect_endpoint(
    file: UploadFile = File(...),
    authorization: str = Header(default=""),
):
    _require_auth(authorization)
    import torch

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    t0 = time.time()
    waveform, sr = _load_audio_to_tensor(data, _STATE["sample_rate"])
    duration = float(waveform.shape[-1]) / sr

    with torch.no_grad():
        result, message = _STATE["detector"].detect_watermark(
            waveform.unsqueeze(0), sample_rate=sr,
        )
        # `result` is a (B,) tensor of detection probability.
        # `message` is (B, 16) bits (only meaningful if result is high).
        prob = float(result.squeeze().item())
        bits = message.squeeze(0).tolist() if message is not None else []

    found = prob >= 0.5
    seed = _bits_to_hex(bits) if found else None

    log.info(
        "detect prob=%.3f found=%s duration_sec=%.2f elapsed_ms=%d",
        prob, found, duration, int((time.time() - t0) * 1000),
    )
    return JSONResponse({
        "found": found,
        "seed": seed,
        "confidence": prob,
        "duration_sec": duration,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------------------
# Modal class — holds per-container state, exposes the FastAPI app
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    secrets=[
        modal.Secret.from_name("doseedo-chatbot-gate"),  # provides VLLM_API_KEY
    ],
    min_containers=0,
    scaledown_window=600,  # 10 min — verify traffic is bursty
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=8)
class Watermark:
    @modal.enter(snap=True)
    def _load_models(self):
        # Heavy imports only at boot. The image build pre-pulled the
        # AudioSeal weights so this doesn't hit the network at runtime.
        from audioseal import AudioSeal

        _STATE["generator"] = AudioSeal.load_generator("audioseal_wm_16bits")
        _STATE["detector"] = AudioSeal.load_detector("audioseal_detector_16bits")
        log.info("audioseal generator + detector loaded")

    @modal.enter(snap=False)
    def _load_secrets(self):
        # Secrets are injected post-snapshot to avoid baking them into the
        # checkpoint.
        _STATE["api_key"] = os.environ.get("VLLM_API_KEY") or os.environ.get(
            "VLLM_GATE_TOKEN"
        )
        if not _STATE["api_key"]:
            log.warning(
                "no VLLM_API_KEY/VLLM_GATE_TOKEN — endpoints will return 503"
            )

    @modal.asgi_app()
    def asgi(self):
        return api


# ---------------------------------------------------------------------------
# Helpers exposed to other Modal apps (e.g. stemphonic_server) — call via
# `Watermark().embed_local(...)` from inside the same Modal workspace to
# skip the HTTP hop. The Stemphonic worker imports this module after each
# generation and calls `embed_inline()` to embed the payload before
# uploading the stem to R2.
# ---------------------------------------------------------------------------

def embed_inline(audio_bytes: bytes, payload_hex: str) -> bytes:
    """Synchronous in-process embed. Used by the Stemphonic worker — does
    not require the FastAPI surface to be up. Caller must have run
    `_load_models()` first (the Stemphonic image should pip-install
    audioseal too)."""
    import torch

    if _STATE["generator"] is None:
        from audioseal import AudioSeal

        _STATE["generator"] = AudioSeal.load_generator("audioseal_wm_16bits")
    waveform, sr = _load_audio_to_tensor(audio_bytes, 16000)
    msg = _payload_to_bits(payload_hex)
    with torch.no_grad():
        wm = _STATE["generator"].get_watermark(
            waveform.unsqueeze(0), sample_rate=sr, message=msg,
        )
        out = waveform.unsqueeze(0) + wm
    return _tensor_to_wav(out.squeeze(0), sr)
