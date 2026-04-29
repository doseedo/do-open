"""Watermark + attestation hook for Stemphonic generations.

Called inline from modal/modal_stemphonic.py::_persist_task_outputs for
every audio file we're about to upload to R2. Two side effects:

  1. Embeds an inaudible AudioSeal watermark in the audio bytes at the
     source's native sample rate, per channel. Skipped when the caller
     passes embed=False (paid-tier opt-out) — the file passes through
     untouched but we still register the row so the chain proof exists.
  2. POSTs the registration to /internal/attestations/watermark on the
     Fly auth-service. The registration is BLOCKING with one retry —
     a hard failure raises, the caller treats the generation as failed
     and the user retries. We no longer ship audio for which the chain
     proof can't be produced.

Image deps required (already added to modal_stemphonic.py image build):
    audioseal==0.1.4
    soundfile (already there)
    torch (already there)

Secrets required on the Stemphonic Modal class:
    doseedo-gate
        AUTH_SERVICE_URL
        INTERNAL_SECRET
"""

from __future__ import annotations

import hashlib
import io
import json as _json
import logging
import os
import secrets
import time
import urllib.error
import urllib.request
from typing import Optional, Tuple

log = logging.getLogger("stemphonic.watermark")


# ---------------------------------------------------------------------------
# Lazy-loaded model state. The Stemphonic worker is single-container so
# module-level state is fine.
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
    """Map the first 2 bytes of the seed to AudioSeal-16's 16-bit message."""
    import torch

    raw = bytes.fromhex(payload_hex)[:2]
    bits = []
    for b in raw:
        for i in range(8):
            bits.append(1 if (b >> (7 - i)) & 1 else 0)
    while len(bits) < 16:
        bits.append(0)
    return torch.tensor([bits[:16]], dtype=torch.int32)


def _decode_audio_native(data: bytes):
    """Bytes → (C, T) float32 torch tensor at the source's native sample
    rate. Mono inputs return C=1, stereo C=2."""
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

    if wav.ndim == 1:
        wav = wav.reshape(-1, 1)
    waveform = torch.from_numpy(wav.T.astype("float32"))  # (C, T)
    return waveform, int(sr)


def _encode_tensor_to_wav(waveform, sr: int) -> bytes:
    """Encode (C, T) → WAV bytes at native sr."""
    import soundfile as sf

    arr = waveform.cpu().numpy()
    if arr.ndim == 2 and arr.shape[0] in (1, 2):
        arr = arr.T
    buf = io.BytesIO()
    sf.write(buf, arr, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def _embed_native(waveform, sr: int, msg_bits):
    """Embed per channel at native sr. Returns (C, T) tensor."""
    import torch

    gen = _get_generator()
    out_channels = []
    for ch in range(waveform.shape[0]):
        x = waveform[ch:ch + 1].unsqueeze(0)  # (1, 1, T)
        with torch.no_grad():
            wm = gen.get_watermark(x, sample_rate=sr, message=msg_bits)
            out_channels.append((x + wm).squeeze(0).squeeze(0))
    return torch.stack(out_channels, dim=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fresh_seed_hex() -> str:
    """16 random bytes as 32 hex chars."""
    return secrets.token_hex(16)


def embed_watermark(audio_bytes: bytes, payload_hex: str) -> Optional[bytes]:
    """Embed AudioSeal at native rate / channel count. Returns the
    watermarked WAV bytes or None on failure (caller falls back to the
    original bytes). Embed is best-effort: if it raises, the audio
    still ships, just without the audio-side mark."""
    try:
        waveform, sr = _decode_audio_native(audio_bytes)
        msg = _payload_to_bits(payload_hex)
        out = _embed_native(waveform, sr, msg)
        return _encode_tensor_to_wav(out, sr)
    except Exception as e:
        log.warning("embed_watermark failed: %s — passing original bytes through", e)
        return None


class AttestationRegistrationError(RuntimeError):
    """Raised when /internal/attestations/watermark refused or was
    unreachable after a retry. The Stemphonic worker treats this as a
    hard generation failure rather than shipping an unindexed file."""


def _post_attestation(
    *,
    audio_sha256: str,
    seed_hex: Optional[str],
    generation_id: str,
    user_id: Optional[str],
    tier: str,
    model_version: str,
) -> None:
    """POST to auth-service. Raises AttestationRegistrationError on hard
    failure. One in-line retry on transient errors / 5xx."""
    auth_url = os.environ.get("AUTH_SERVICE_URL", "").rstrip("/")
    secret = os.environ.get("INTERNAL_SECRET", "")
    if not auth_url or not secret:
        raise AttestationRegistrationError(
            "AUTH_SERVICE_URL / INTERNAL_SECRET not configured"
        )

    body = _json.dumps({
        "audio_sha256": audio_sha256,
        "seed": seed_hex,
        "generation_id": generation_id,
        "user_id": int(user_id) if user_id and str(user_id).isdigit() else None,
        "tier": tier or "unknown",
        "model_version": model_version or "stemphonic",
    }).encode("utf-8")

    last_err: Optional[str] = None
    for attempt in (1, 2):
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
            with urllib.request.urlopen(req, timeout=10) as resp:
                if 200 <= resp.status < 300:
                    return
                last_err = f"HTTP {resp.status}"
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}"
            # 4xx is terminal — bad payload, no point retrying.
            if 400 <= e.code < 500:
                raise AttestationRegistrationError(last_err) from e
        except Exception as e:
            last_err = str(e)
        if attempt == 1:
            time.sleep(1.0)
    raise AttestationRegistrationError(last_err or "unknown")


def embed_and_attest(
    audio_bytes: bytes,
    *,
    generation_id: str,
    user_id: Optional[str],
    tier: str = "unknown",
    model_version: str = "stemphonic",
    embed: bool = True,
) -> Tuple[bytes, str, Optional[str]]:
    """One-shot: embed (if requested), hash the bytes the user will
    receive, and register the attestation. Returns (audio_bytes,
    audio_sha256, seed_hex_or_none).

    Embed is best-effort; registration is required. If registration
    fails the caller should treat the generation as failed and not
    upload the file — the user retries and the next attempt registers
    cleanly. This is the inverse of the previous best-effort posture
    but is what the public /verify route now requires (every shipped
    file must have a row + chain proof).
    """
    seed: Optional[str] = None
    out_bytes = audio_bytes
    if embed:
        seed = fresh_seed_hex()
        watermarked = embed_watermark(audio_bytes, seed)
        if watermarked is not None:
            out_bytes = watermarked
        else:
            # Embed failed. Register without the audio-side mark — the
            # chain proof still binds to audio_sha256 of the original
            # bytes, which is what the user receives.
            seed = None

    audio_sha256 = hashlib.sha256(out_bytes).hexdigest()
    _post_attestation(
        audio_sha256=audio_sha256,
        seed_hex=seed,
        generation_id=generation_id,
        user_id=user_id,
        tier=tier,
        model_version=model_version,
    )
    return out_bytes, audio_sha256, seed
