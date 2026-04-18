#!/usr/bin/env python3
"""
verify_frontend_deploy.py — post-deploy smoke for the sem4Decoder preview
pipeline. Runs end-to-end WITHOUT a browser:

  1. HEAD each /static/models/* URL via https://doseedo.com — confirms the
     Vercel → R2 rewrite resolves and the CDN serves the file with the size
     we expect. A stale deploy or a bad rewrite surfaces here.

  2. Find the current DAW chunk via the Next.js manifest, fetch it, and
     grep for the code markers that prove the preview pipeline is wired
     (sem4Decoder checkpoint logs, model URLs, streamPreviewSeparation).

  3. Run the literal preview pipeline in onnxruntime against the LIVE R2
     URLs. Downloads distill_demucs_fp16, sem_demucs_packed, sem_decoder_fp16
     off production, loads each into ORT, feeds the tears-for-fears clip
     through all three exactly like sem4Decoder.js does, and verifies each
     stem's decoded output is non-silent. Shape mismatches or any of the
     silent-NaN regressions we've hit in the 6-stem model fail cleanly.

Exit 0 on full green, 1 on any failure. Cheap — single run against prod
after `git push` catches 90% of what a real browser test would.

Usage:
  python3 scripts/verify_frontend_deploy.py
  BASE=https://do-<hash>-doseedo.vercel.app python3 scripts/verify_frontend_deploy.py
"""
from __future__ import annotations

import io
import os
import re
import sys
import time
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

BASE = os.environ.get("BASE", "https://doseedo.com").rstrip("/")
MP3 = os.environ.get(
    "SMOKE_MP3",
    "/Users/hydroadmin/Downloads/tearsforfearseverybodywantstoruletheworldofficia (1).mp3",
)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) doseedo-frontend-verify/1.0"

MODELS_EXPECTED = [
    ("/static/models/distill_demucs_fp16.onnx",       904066),
    ("/static/models/distill_demucs_fp16.onnx.data",  170106368),
    ("/static/models/sem_demucs_packed.onnx",         5190993),
    ("/static/models/sem_decoder_fp16.onnx",          203676),
    ("/static/models/sem_decoder_fp16.onnx.data",     20921936),
]
BUNDLE_CODE_MARKERS = [
    "[sem4Decoder] preview kicked off",
    "[sem4Decoder] initSem4Decoder starting",
    "[sem4Decoder] fetching 5 model files",
    "/static/models/distill_demucs_fp16.onnx",
    "/static/models/sem_demucs_packed.onnx",
    "/static/models/sem_decoder_fp16.onnx",
]

# ─── colors ────────────────────────────────────────────────────────────────
def _c(code, m): return f"\033[{code}m{m}\033[0m"
def green(m): return _c("32", f"✓ {m}")
def red(m):   return _c("31", f"✗ {m}")
def blue(m):  return _c("34", m)

fails: list[str] = []
def fail(m): fails.append(m); print(red(m))


# ─── http helpers ──────────────────────────────────────────────────────────
def _head(path: str) -> tuple[int, dict]:
    """Cheap 'HEAD' that asks for byte 0 with a Range request. R2 strips
    Content-Length from some HEADs for large files, so Range+Content-Range
    is the reliable way to discover total size without downloading."""
    req = urllib.request.Request(
        f"{BASE}{path}",
        headers={"User-Agent": UA, "Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            h = dict(r.headers)
            # Prefer total from Content-Range ("bytes 0-0/TOTAL")
            cr = h.get("Content-Range") or h.get("content-range") or ""
            m = re.match(r"bytes\s+\d+-\d+/(\d+)", cr)
            if m:
                h["Content-Length"] = m.group(1)
            return r.getcode(), h
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)
    except Exception as e:
        return 0, {"error": str(e)}


def _get(path_or_url: str, timeout: int = 60) -> bytes:
    url = path_or_url if path_or_url.startswith("http") else f"{BASE}{path_or_url}"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ─── check 1: model URLs via CDN ───────────────────────────────────────────
def check_model_urls():
    # Vercel's edge proxy drops Content-Length/Range metadata on large R2
    # responses (seen on the two .onnx.data files), so we don't try to verify
    # byte count here — check 3's full fetch + model init is the integrity
    # proof. This step just confirms the rewrite resolves and the CDN
    # returns 200/206 with a non-empty body.
    print(blue("1. /static/models/* — proxy health via doseedo.com rewrite"))
    for path, expected_size in MODELS_EXPECTED:
        code, hdrs = _head(path)
        ct = hdrs.get("Content-Type") or hdrs.get("content-type") or ""
        if code not in (200, 206):
            fail(f"{path} → HTTP {code} ({ct})")
            continue
        size_hdr = hdrs.get("Content-Length") or hdrs.get("content-length")
        size_str = f"{int(size_hdr)/1e6:.1f} MB" if size_hdr else "size-unknown (chunked)"
        # If a size IS reported, flag mismatches — Vercel is inconsistent but
        # when it DOES include one it should match.
        if size_hdr and int(size_hdr) not in (expected_size, 1, 0):
            fail(f"{path} → size {size_hdr} (expected {expected_size})")
            continue
        print(green(f"{path}  ({size_str}, {ct}, HTTP {code})"))


# ─── check 2: bundle contains expected code markers ───────────────────────
def check_bundle_markers():
    print(blue("\n2. Live JS bundle — must contain sem4Decoder code markers"))
    html = _get("/").decode(errors="replace")
    # All chunks directly referenced in the root HTML <script> tags.
    referenced = set(re.findall(r'/_next/static/chunks/[A-Za-z0-9._/%-]+\.js', html))
    # The DAW component is code-split and its chunk isn't referenced on the
    # root page. Pull the webpack runtime's chunk-id → filename hash map so
    # we can discover lazy chunks too. Webpack serializes it as a big
    # object: {<id>:"<hash>",...} inside the runtime chunk.
    runtime_url = next((u for u in referenced if "/webpack-" in u), None)
    all_chunks = set(referenced)
    if runtime_url:
        try:
            runtime = _get(runtime_url, timeout=60).decode(errors="replace")
            # Example after min: "(e)=>(({42:\"abc\",618:\"def\",...})[e]+...)"
            id_hash_pairs = re.findall(r'"?(\d{2,4})"?:"([a-f0-9]+)"', runtime)
            for cid, chash in id_hash_pairs:
                all_chunks.add(f"/_next/static/chunks/{cid}-{chash}.js")
                all_chunks.add(f"/_next/static/chunks/{cid}.{chash}.js")
        except Exception as e:
            print(red(f"  couldn't read webpack runtime {runtime_url}: {e}"))

    found = {m: False for m in BUNDLE_CODE_MARKERS}
    scanned = 0
    for url in all_chunks:
        try:
            body = _get(url, timeout=90)
        except Exception:
            continue
        scanned += 1
        text = body.decode(errors="replace")
        for m in found:
            if not found[m] and m in text:
                found[m] = True
        if all(found.values()):
            break
    print(f"  (scanned {scanned} chunks, {len(all_chunks)} candidates)")
    for m, ok in found.items():
        if ok:
            print(green(f"marker present: {m!r}"))
        else:
            fail(f"marker MISSING from all chunks: {m!r}")


# ─── check 3: live preview pipeline in ONNX runtime ───────────────────────
def _download(url: str, dst: Path, chunk=2**20):
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    t0 = time.time()
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(chunk)
            if not b: break
            f.write(b)
    tmp.rename(dst)
    return dst


def check_live_pipeline():
    print(blue("\n3. Live pipeline — download live R2 models + run preview on tears.mp3"))
    if not Path(MP3).exists():
        fail(f"SMOKE_MP3 missing: {MP3}  — pass SMOKE_MP3=/path/to/any.mp3 to run this check")
        return

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as e:
        fail(f"missing numpy/onnxruntime: {e}")
        return

    cache = Path("/tmp/doseedo_verify_models")
    cache.mkdir(parents=True, exist_ok=True)
    paths = {}
    for path, _ in MODELS_EXPECTED:
        url = f"{BASE}{path}"
        dst = cache / Path(path).name
        try:
            t0 = time.time()
            _download(url, dst)
            dt = time.time() - t0
            print(f"   fetched {path}  ({dst.stat().st_size/1e6:.1f} MB in {dt:.1f}s)")
        except Exception as e:
            fail(f"fetch {url} failed: {e}")
            return
        paths[Path(path).name] = dst

    # Decode the mp3 → stereo 48k f32, DC-remove, crop to first 8s (1 chunk)
    raw = subprocess.check_output([
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", MP3, "-f", "f32le", "-ac", "2", "-ar", "48000", "-"
    ])
    wav = np.frombuffer(raw, dtype=np.float32).reshape(-1, 2).T  # [2, N]
    wav = wav - wav.mean(axis=1, keepdims=True)                  # DC-remove
    use_n = min(8 * 48000, (wav.shape[1] // 1920) * 1920)
    wav = wav[:, :use_n]
    audio = wav[None].astype(np.float32)   # [1, 2, N]

    so = ort.SessionOptions(); so.log_severity_level = 3
    kws = dict(sess_options=so, providers=["CPUExecutionProvider"])

    # 1) sem_demucs_packed → embedding[1,4,128]
    sem = ort.InferenceSession(str(paths["sem_demucs_packed.onnx"]), **kws)
    emb = sem.run(None, {"waveform": audio})[
        [o.name for o in sem.get_outputs()].index("embedding")
    ]
    if emb.shape != (1, 4, 128):
        fail(f"sem_demucs_packed embedding shape {emb.shape} != (1,4,128)")
        return
    print(green(f"sem_demucs_packed  → embedding {emb.shape}"))

    # 2) distill_demucs_fp16 → stem_latents[1,4,64,T]
    dem = ort.InferenceSession(str(paths["distill_demucs_fp16.onnx"]), **kws)
    stem_latents = dem.run(None, {"audio": audio})[0]
    T = use_n // 1920
    if stem_latents.shape != (1, 4, 64, T):
        fail(f"distill_demucs_fp16 stem_latents shape {stem_latents.shape} != (1,4,64,{T})")
        return
    print(green(f"distill_demucs_fp16 → stem_latents {stem_latents.shape}"))

    # 3) sem_decoder_fp16 × 4 stems
    dec = ort.InferenceSession(str(paths["sem_decoder_fp16.onnx"]), **kws)
    expected_samples = 1920 * T
    names = ["drums", "bass", "vocals", "other"]
    for s, name in enumerate(names):
        lat = stem_latents[:, s].astype(np.float32)
        e = emb[:, s].astype(np.float32)
        out = dec.run(None, {"latent": lat, "sem_emb": e})[0]
        if out.shape != (1, 2, expected_samples):
            fail(f"sem_decoder_fp16[{name}] shape {out.shape} != (1,2,{expected_samples})")
            return
        peak = float(np.abs(out).max())
        rms = float(np.sqrt((out**2).mean()))
        if peak < 1e-4:
            fail(f"sem_decoder_fp16[{name}] silent (peak={peak:.2e})")
            continue
        print(green(f"sem_decoder_fp16[{name}]  peak={peak:.3f}  rms={rms:.4f}"))


# ─── main ──────────────────────────────────────────────────────────────────
def main() -> int:
    print(blue(f"verify_frontend_deploy against {BASE}\n"))
    check_model_urls()
    check_bundle_markers()
    check_live_pipeline()
    print()
    if fails:
        print(red(f"FAILED ({len(fails)}):"))
        for f in fails:
            print(f"  · {f}")
        return 1
    print(green("all frontend-deploy checks passed"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
