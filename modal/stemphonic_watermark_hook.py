"""Watermark + attestation hook for Stemphonic generations.

Called inline from modal/modal_stemphonic.py::_persist_task_outputs for
every audio file we're about to upload to R2. Two side effects:

  1. Embeds an inaudible AudioSeal watermark in the audio bytes. The
     watermark seed is fresh per file (16 random bytes, hex).
  2. POSTs the seed + metadata to the Fly auth-service at
     /internal/attestations/watermark. Auth-service stores the row and
     the existing provenance_publisher worker submits it to the
     deployed Polygon contract (commitRecord) in its next tick.

Both steps are best-effort — if either fails the original audio bytes
flow through unchanged and a warning is logged. We never block a
generation on watermarking.

Image deps required (already added to modal_stemphonic.py image build):
    audioseal==0.1.4
    soundfile (already there)
    torch (already there)

Secrets required on the Stemphonic Modal class:
    doseedo-gate
        AUTH_SERVICE_URL    (already required by the gate code path)
        INTERNAL_SECRET     (already required by the gate code path)

No DB driver, no DATABASE_URL — the hook only POSTs over HTTP. The
auth-service owns the DB.
"""

from __future__ import annotations

import io
import json as _json
import logging
import os
import secrets
import urllib.error
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger("stemphonic.watermark")


# ---------------------------------------------------------------------------
# Lazy-loaded model state — first call pays the AudioSeal load cost (~1s on
# CPU). Subsequent calls reuse. The Stemphonic worker is single-container
# so module-level state is fine.
# ---------------------------------------------------------------------------

_GENERATOR = None


def _get_generator():
    global _GENERATOR
    if _GENERATOR is not None:
        return _GENERATOR
    from audioseal import AudioSeal
    _GENERATOR = AudioSeal.load_generator("audioseal_wm_16bits")
    log.info("audioseal generator loaded for stemphonic watermark hook")
    return _GENERATOR


def _payload_to_bits(payload_hex: str):
    """Map the first 2 bytes of the seed onto the 16-bit AudioSeal message."""
    import torch

    raw = bytes.fromhex(payload_hex)[:2]
    bits = []
    for b in raw:
        for i in range(8):
            bits.append(1 if (b >> (7 - i)) & 1 else 0)
    while len(bits) < 16:
        bits.append(0)
    return torch.tensor([bits[:16]], dtype=torch.int32)


def _decode_audio_to_tensor(data: bytes, target_sr: int = 16000):
    """Bytes → mono float32 torch tensor at target_sr. Punts to torchaudio
    if soundfile can't read the format (mp3/opus)."""
    import numpy as np
    import soundfile as sf
    import torch

    try:
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception:
        import tempfile
        import torchaudio

        with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as fh:
            fh.write(data)
            fh.flush()
            tensor, sr = torchaudio.load(fh.name)
            wav = tensor.numpy().T

    if wav.ndim == 2 and wav.shape[1] > 1:
        wav = wav.mean(axis=1, keepdims=True)
    elif wav.ndim == 1:
        wav = wav.reshape(-1, 1)
    waveform = torch.from_numpy(wav.T.astype("float32"))

    if sr != target_sr:
        import torchaudio.functional as taF
        waveform = taF.resample(waveform, sr, target_sr)
    return waveform, target_sr


def _encode_tensor_to_wav(waveform, sr: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    sf.write(
        buf,
        waveform.squeeze(0).cpu().numpy(),
        sr,
        subtype="PCM_16",
        format="WAV",
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed_watermark(audio_bytes: bytes, payload_hex: str) -> Optional[bytes]:
    """Returns watermarked WAV bytes, or None on any failure (caller
    should fall back to the original bytes). Resampled to 16 kHz mono —
    the Opus encoder downstream lossy-encodes anyway, so the rate change
    is not an additional quality penalty."""
    try:
        import torch

        gen = _get_generator()
        waveform, sr = _decode_audio_to_tensor(audio_bytes, target_sr=16000)
        msg = _payload_to_bits(payload_hex)
        with torch.no_grad():
            wm = gen.get_watermark(
                waveform.unsqueeze(0), sample_rate=sr, message=msg,
            )
            out = waveform.unsqueeze(0) + wm
        return _encode_tensor_to_wav(out.squeeze(0), sr)
    except Exception as e:
        log.warning("embed_watermark failed: %s — passing original bytes through", e)
        return None


def _post_attestation(
    *,
    seed_hex: str,
    generation_id: str,
    user_id: Optional[str],
    tier: str,
    model_version: str,
    wallet: Optional[str] = None,
) -> bool:
    """POST to auth-service /internal/attestations/watermark. Best-effort.

    Auth-service owns the DB write + the Polygon publish lifecycle; we
    just hand off the payload here and trust the server. Returns True
    on 2xx, False on anything else. Never raises."""
    auth_url = os.environ.get("AUTH_SERVICE_URL", "").rstrip("/")
    secret = os.environ.get("INTERNAL_SECRET", "")
    if not auth_url or not secret:
        log.warning(
            "AUTH_SERVICE_URL / INTERNAL_SECRET not configured — "
            "skipping attestation registration"
        )
        return False

    body = _json.dumps({
        "seed": seed_hex,
        "generation_id": generation_id,
        "user_id": int(user_id) if user_id and str(user_id).isdigit() else None,
        "tier": tier or "unknown",
        "model_version": model_version or "stemphonic",
        "wallet": wallet,
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{auth_url}/internal/attestations/watermark",
        data=body,
        headers={
            "Content-Type": "application/json",
            "X-Internal-Secret": secret,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except urllib.error.HTTPError as e:
        log.warning(
            "attestation POST HTTP %d for seed=%s gen=%s",
            e.code, seed_hex[:8], generation_id,
        )
        return False
    except Exception as e:
        log.warning(
            "attestation POST unreachable for seed=%s gen=%s: %s",
            seed_hex[:8], generation_id, e,
        )
        return False


def fresh_seed_hex() -> str:
    """16 random bytes as 32 hex chars."""
    return secrets.token_hex(16)


def embed_and_attest(
    audio_bytes: bytes,
    *,
    generation_id: str,
    user_id: Optional[str],
    tier: str = "unknown",
    model_version: str = "stemphonic",
    wallet: Optional[str] = None,
) -> Tuple[bytes, Optional[str]]:
    """One-shot: embed a fresh watermark and register the attestation.
    Returns (audio_bytes, seed_hex). On any failure we fall back to the
    *original* bytes and a None seed so the caller can still upload the
    file — the user-visible flow never depends on this hook succeeding."""
    seed = fresh_seed_hex()
    watermarked = embed_watermark(audio_bytes, seed)
    if watermarked is None:
        return audio_bytes, None
    posted = _post_attestation(
        seed_hex=seed,
        generation_id=generation_id,
        user_id=user_id,
        tier=tier,
        model_version=model_version,
        wallet=wallet,
    )
    if not posted:
        log.info(
            "watermark embedded but attestation registration failed (seed=%s gen=%s) — "
            "verifier will surface as 'unindexed'",
            seed[:8], generation_id,
        )
    return watermarked, seed
