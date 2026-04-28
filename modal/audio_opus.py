"""modal/audio_opus.py — ffmpeg-backed Opus encoder for Modal-side persistence.

Mirrors logic_engine/audio_opus.py in the desktop tree, trimmed to what the
Modal container needs (encode-from-file → bytes). The Modal image already
ships ffmpeg via apt_install("ffmpeg"), so no path probing.

Defaults match the rest of the system:
  bitrate 128 kbps, 48 kHz, OGG container, application "audio".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

DEFAULT_BITRATE_KBPS = 128
DEFAULT_SAMPLE_RATE = 48_000


def is_available() -> bool:
    return shutil.which("ffmpeg") is not None


def encode_file(
    audio_path: str,
    *,
    bitrate_kbps: int = DEFAULT_BITRATE_KBPS,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
) -> bytes:
    """Encode an on-disk audio file to Opus bytes (OGG container).

    Raises RuntimeError if ffmpeg is missing or fails."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found — cannot encode to Opus")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(audio_path)

    fd, tmp_path = tempfile.mkstemp(prefix="doo_opus_", suffix=".opus")
    os.close(fd)
    try:
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", audio_path,
            "-c:a", "libopus",
            "-b:a", f"{bitrate_kbps}k",
            "-ar", str(sample_rate),
            "-vn",
            "-application", "audio",
            tmp_path,
        ]
        res = subprocess.run(cmd, capture_output=True)
        if res.returncode != 0:
            raise RuntimeError(
                f"ffmpeg encode failed ({res.returncode}): "
                f"{res.stderr.decode('utf-8', 'replace').strip()[:500]}"
            )
        with open(tmp_path, "rb") as fh:
            return fh.read()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
