"""Doseedo audio watermark service — CPU container running AudioSeal.

Endpoints (all bearer-gated against VLLM_API_KEY):

  POST /embed    multipart/form-data { file: audio, payload: 32-hex }
                 Returns: audio/wav body with the watermark embedded at
                 the source's native sample rate, stereo when the input
                 is stereo. Mono inputs stay mono.
                 Headers: x-doseedo-seed, x-doseedo-sr, x-doseedo-sha256.

  POST /detect   multipart/form-data { file: audio }
                 Returns JSON:
                   { "found": bool,
                     "confidence": 0.0–1.0,
                     "audio_sha256": "<64-hex>",   // sha256 of the upload
                     "duration_sec": float,
                     "scanned_at": ISO-8601 }
                 NOTE: we no longer return a seed prefix — the audio
                 carrier is a 16-bit signal that cannot uniquely
                 identify a registered row. The verifier hashes the
                 uploaded bytes and looks the row up by audio_sha256.

  GET  /health   public.

Why native sample rate / stereo:
    AudioSeal's `get_watermark` accepts (B, C, T) at any sample rate the
    generator supports (it resamples internally to its 16 kHz training
    rate but writes the additive watermark back at the input rate). We
    keep the user's audio at the source rate to avoid a hard quality
    regression on every paid-tier export.
"""

import hashlib
import io
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import modal

# FastAPI imports MUST be at module scope (Pydantic v2 needs UploadFile
# in globalns to resolve annotations on FastAPI route registration).
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
        raise HTTPException(status_code=503, detail="auth not configured")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(status_code=401, detail="bad bearer")


def _payload_to_bits(payload_hex: str):
    """Map the first 2 bytes of the 32-hex seed to AudioSeal-16's 16-bit
    message. The full seed is stored server-side; the on-audio carrier
    only conveys 16 bits and is treated as a 'doseedo signal' classifier
    — uniqueness comes from audio_sha256, not this prefix."""
    import torch

    if len(payload_hex) < 4:
        raise HTTPException(status_code=400, detail="payload too short")
    raw = bytes.fromhex(payload_hex)[:2]
    bits = []
    for b in raw:
        for i in range(8):
            bits.append(1 if (b >> (7 - i)) & 1 else 0)
    while len(bits) < 16:
        bits.append(0)
    return torch.tensor([bits[:16]], dtype=torch.int32)


def _load_audio_native(data: bytes):
    """Decode arbitrary audio bytes preserving native sample rate AND
    channel count. Returns (waveform[C, T], sr) as a torch float32
    tensor. Mono inputs come back with C=1; stereo with C=2."""
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
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

    if wav.ndim == 1:
        wav = wav.reshape(-1, 1)
    waveform = torch.from_numpy(wav.T.astype("float32"))  # (C, T)
    return waveform, int(sr)


def _tensor_to_wav(waveform, sr: int) -> bytes:
    """Encode a (C, T) tensor to WAV bytes at native sr."""
    import soundfile as sf

    arr = waveform.cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] in (1, 2):
        arr = arr.T  # soundfile wants (T, C)
    elif arr.ndim == 1:
        pass  # mono is fine as-is
    buf = io.BytesIO()
    sf.write(buf, arr, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _embed_native(waveform, sr: int, msg_bits) -> "torch.Tensor":
    """Embed the watermark per channel at the native sample rate.
    AudioSeal's generator accepts (B, 1, T); we run it once per channel
    and stack so stereo stays stereo. Returns the watermarked tensor
    shaped (C, T) at sr."""
    import torch

    gen = _STATE["generator"]
    out_channels = []
    for ch in range(waveform.shape[0]):
        x = waveform[ch:ch + 1].unsqueeze(0)  # (1, 1, T)
        wm = gen.get_watermark(x, sample_rate=sr, message=msg_bits)
        out_channels.append((x + wm).squeeze(0).squeeze(0))
    return torch.stack(out_channels, dim=0)  # (C, T)


@api.get("/health")
async def health():
    return JSONResponse({
        "ok": _STATE["generator"] is not None and _STATE["detector"] is not None,
        "model": "audioseal_wm_16bits / audioseal_detector_16bits",
        "embed_mode": "native_rate_stereo",
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

    waveform, sr = _load_audio_native(data)
    msg_bits = _payload_to_bits(payload)
    with torch.no_grad():
        watermarked = _embed_native(waveform, sr, msg_bits)

    body = _tensor_to_wav(watermarked, sr)
    sha = hashlib.sha256(body).hexdigest()
    log.info(
        "embed ok bytes_in=%d bytes_out=%d sr=%d ch=%d payload=%s sha=%s",
        len(data), len(body), sr, waveform.shape[0], payload[:16], sha[:12],
    )
    return Response(
        content=body,
        media_type="audio/wav",
        headers={
            "x-doseedo-seed": payload,
            "x-doseedo-sr": str(sr),
            "x-doseedo-sha256": sha,
        },
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

    audio_sha256 = hashlib.sha256(data).hexdigest()

    t0 = time.time()
    waveform, sr = _load_audio_native(data)
    duration = float(waveform.shape[-1]) / sr

    # Detector takes (B, 1, T) mono. Mix to mono for the classifier;
    # this does not affect the audio we return — we don't return audio.
    mono = waveform.mean(dim=0, keepdim=True).unsqueeze(0)  # (1, 1, T)
    with torch.no_grad():
        result, _message = _STATE["detector"].detect_watermark(
            mono, sample_rate=sr,
        )
        if hasattr(result, "squeeze"):
            sq = result.squeeze()
            prob = float(sq.item() if hasattr(sq, "item") else sq)
        else:
            prob = float(result)

    found = prob >= 0.5

    log.info(
        "detect prob=%.3f found=%s sha=%s duration_sec=%.2f elapsed_ms=%d",
        prob, found, audio_sha256[:12], duration, int((time.time() - t0) * 1000),
    )
    return JSONResponse({
        "found": found,
        "confidence": prob,
        "audio_sha256": audio_sha256,
        "duration_sec": duration,
        "scanned_at": datetime.now(timezone.utc).isoformat(),
    })


# ---------------------------------------------------------------------------
# Modal class
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    secrets=[
        modal.Secret.from_name("doseedo-chatbot-gate"),  # provides VLLM_API_KEY
    ],
    min_containers=0,
    scaledown_window=600,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=8)
class Watermark:
    @modal.enter(snap=True)
    def _load_models(self):
        from audioseal import AudioSeal

        _STATE["generator"] = AudioSeal.load_generator("audioseal_wm_16bits")
        _STATE["detector"] = AudioSeal.load_detector("audioseal_detector_16bits")
        log.info("audioseal generator + detector loaded")

    @modal.enter(snap=False)
    def _load_secrets(self):
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
