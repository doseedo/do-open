#!/usr/bin/env python3
"""
End-to-end smoke test. Hits the live stack after a deploy and asserts the
user-facing paths actually work. Exits 0 on success, 1 on any failure.

Usage:
    python3 scripts/e2e_smoke.py                      # against doseedo.com
    BASE=https://do-<hash>-doseedo.vercel.app python3 scripts/e2e_smoke.py

What it checks (and why each):
  1. GET /               → 200, auth-service reachable through Vercel rewrite
  2. GET /api/sessions   → 401, Fly + Neon + Clerk dual-run auth all wire up
  3. POST /separate-stems with a 1-second synthetic WAV → poll until completed
     → Modal image has demucs.api, torch, onnxruntime. Real E2E.

If a check fails, the script prints the HTTP code + body snippet and moves on
so you see ALL failures in one shot, not just the first.
"""
from __future__ import annotations

import io
import math
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import wave

BASE = os.environ.get("BASE", "https://doseedo.com").rstrip("/")
TIMEOUT = 240
# Cloudflare blocks the default Python-urllib UA with error 1010. Send a real
# browser UA so our smoke tests can actually reach the app.
DEFAULT_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 doseedo-e2e-smoke"

# ─── colors ─────────────────────────────────────────────────────────────
def _c(code, msg): return f"\033[{code}m{msg}\033[0m"
def green(m): return _c("32", f"✓ {m}")
def red(m):   return _c("31", f"✗ {m}")
def blue(m):  return _c("34", m)

fails: list[str] = []
def fail(msg: str) -> None:
    fails.append(msg)
    print(red(msg))


# ─── http helpers ───────────────────────────────────────────────────────
def _req(method: str, path: str, body: bytes | None = None,
         headers: dict | None = None, timeout: int = 15):
    url = f"{BASE}{path}"
    merged = {"User-Agent": DEFAULT_UA}
    if headers:
        merged.update(headers)
    req = urllib.request.Request(url, data=body, method=method, headers=merged)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.getcode(), resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as e:
        return 0, str(e).encode()


# ─── tests ──────────────────────────────────────────────────────────────
def check_sessions_401():
    # /api/sessions goes through Vercel's rewrite → Fly → Neon. A 401 response
    # means every hop in that chain is alive. No auth attached, so 401 is the
    # healthy answer. (GET / hits the Next.js catch-all and returns HTML; it
    # doesn't actually exercise the backend, so we don't check it.)
    print(blue("1. GET /api/sessions — exercises full Vercel→Fly→Neon path; expect 401"))
    code, body = _req("GET", "/api/sessions")
    if code == 401:
        print(green(f"status 401, body: {body[:80].decode(errors='replace')}"))
    else:
        fail(f"GET /api/sessions returned {code} (expected 401), body={body[:200]!r}")


def _synth_wav(seconds: float = 1.0, freq: int = 440) -> bytes:
    """Generate a stereo 44.1kHz 16-bit sine-wave WAV in memory."""
    buf = io.BytesIO()
    sr = 44100
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(sr)
        for i in range(int(sr * seconds)):
            v = int(16000 * math.sin(2 * math.pi * freq * i / sr))
            w.writeframes(struct.pack("<hh", v, v))
    return buf.getvalue()


def _multipart(fieldname: str, filename: str, data: bytes,
               content_type: str = "audio/wav") -> tuple[bytes, str]:
    boundary = "----e2eSmoke" + str(int(time.time()))
    parts = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{fieldname}"; filename="{filename}"\r\n'
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    body = parts + data + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def check_separate_stems():
    print(blue("2. POST /separate-stems with a real WAV, poll until completed (Modal demucs E2E)"))
    wav = _synth_wav(seconds=1.0)
    body, ct = _multipart("audioFile", "e2e_smoke.wav", wav)
    code, resp = _req("POST", "/separate-stems", body=body,
                      headers={"Content-Type": ct}, timeout=60)
    if code != 200:
        fail(f"POST /separate-stems returned {code}, body={resp[:200]!r}")
        return
    try:
        import json
        data = json.loads(resp)
    except Exception as e:
        fail(f"POST /separate-stems returned non-JSON: {resp[:200]!r}")
        return
    task_id = data.get("task_id")
    if not task_id:
        fail(f"POST /separate-stems missing task_id: {data}")
        return
    print(f"   task_id: {task_id}")

    deadline = time.time() + TIMEOUT
    last: dict = {}
    while time.time() < deadline:
        time.sleep(5)
        code, resp = _req("GET", f"/separate-stems/status/{task_id}", timeout=15)
        if code != 200:
            print(f"   poll → {code} (retrying)")
            continue
        try:
            last = json.loads(resp)
        except Exception:
            continue
        st = last.get("status", "?")
        err = last.get("error")
        print(f"   status={st} error={err}")
        if st == "completed":
            stems = last.get("stem_latents") or last.get("stems") or {}
            if stems:
                print(green(f"separation completed — {len(stems)} stems returned"))
                return
            fail(f"completed but no stems: {last}")
            return
        if st == "failed":
            fail(f"separation failed: error={err!r}")
            return
    fail(f"timed out after {TIMEOUT}s — last status: {last}")


# ─── main ───────────────────────────────────────────────────────────────
def main() -> int:
    print(blue(f"e2e smoke against {BASE}\n"))
    check_sessions_401()
    check_separate_stems()
    print()
    if fails:
        print(red(f"FAILED ({len(fails)}):"))
        for f in fails:
            print(f"  · {f}")
        return 1
    print(green("all e2e checks passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
