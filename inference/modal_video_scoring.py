"""Doseedo video-scoring service — CPU container running the
`video_scoring` package.

Wraps the local-only side of the stack (PySceneDetect + OpenCV +
FilmScoringEngine) and calls the existing chatbot vision endpoint over
HTTP for Moondream queries. CPU-only (no GPU) — Moondream lives in
modal_chatbot.py.

Two endpoints on the same container:

  POST /score    — Bearer-gated. multipart/form-data with field "file".
                   Returns Server-Sent Events:
                     event: shots       data: {"count": N}
                     event: scene       data: {"i":k,"of":N,"start":s,"end":e}
                     event: scene_done  data: {"i":k,"of":N,"mood":...,"tension":...}
                     event: midi        data: {}
                     event: done        data: {scene_data, scene_changes,
                                                duration, midi_base64}
                     event: error       data: {"message":"..."}
                   Optional form fields: bpm (int, default 120),
                   base_progression ('Cm:0,Fm:4,...'), frames_per_scene (int).

  GET  /health   — public.

Auth: same VLLM_API_KEY bearer the chatbot uses.

Deploy:  modal deploy modal/modal_video_scoring.py

Why FastAPI is at MODULE scope (not inside the asgi_app method):
FastAPI builds Pydantic v2 TypeAdapters at endpoint registration time,
and resolves annotations via `typing.get_type_hints()`. When the route
function is defined inside a class method (`def score(self): def score_endpoint(...)`),
get_type_hints does NOT see the enclosing-method locals as part of its
namespace, so `UploadFile` resolves as a bare ForwardRef and pydantic
raises `class-not-fully-defined` on the first request. Defining the
route at module scope where `UploadFile` is in `globalns` fixes it.
The class only needs to hold per-container state; the FastAPI app is a
plain module-level object the class returns from its asgi_app hook.
"""

import asyncio
import base64
import json
import logging
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import modal

# FastAPI imports MUST be at module scope (see header). Same with the
# annotated route function below.
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install_from_requirements("video_scoring/modal_requirements.txt")
    .add_local_dir("video_scoring", remote_path="/root/video_scoring")
)

app = modal.App("doseedo-video-scoring")


# ---------------------------------------------------------------------------
# Per-container state populated at @modal.enter time
# ---------------------------------------------------------------------------
#
# The route handler is module-level (so FastAPI can resolve UploadFile),
# so it reads its dependencies (auth key, vision URL) from this dict
# rather than from `self`. The class fills this in setup_post_snap().

_STATE: dict = {
    "api_key": None,
    "vision_url": None,
}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG = logging.getLogger("video_scoring")
if not _LOG.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(logging.Formatter("%(message)s"))
    _LOG.addHandler(_h)
    _LOG.setLevel(logging.INFO)


def _log_json(tag: str, **fields):
    """Emit a single-line JSON record. Picked up by Modal's log shipper
    and matches the chatbot's tag-keyed structure for Sentry parsing."""
    try:
        _LOG.info(json.dumps({"tag": tag, **fields}, default=str))
    except Exception:
        _LOG.info("tag=%s fields=%r", tag, fields)


# ---------------------------------------------------------------------------
# FastAPI app + routes (module scope)
# ---------------------------------------------------------------------------

api = FastAPI(title="doseedo-video-scoring", version="1.1.0")


def _require_auth(authorization: str) -> None:
    expected = _STATE.get("api_key")
    if not expected:
        # State not yet populated — should never happen for a real request
        # but covers the warm-up path.
        raise HTTPException(status_code=503, detail="service not ready")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer")
    if authorization[len("Bearer "):] != expected:
        raise HTTPException(status_code=401, detail="invalid key")


def _parse_progression(spec: Optional[str]):
    if not spec:
        return None
    out = {}
    for item in spec.split(","):
        if ":" not in item:
            continue
        chord, beat = item.split(":", 1)
        try:
            out[int(beat.strip())] = chord.strip()
        except ValueError:
            continue
    return out or None


@api.get("/health")
async def health():
    try:
        import scenedetect  # noqa: F401
        ok = True
    except ImportError:
        ok = False
    return JSONResponse({
        "ok": ok,
        "scenedetect": ok,
        "vision_origin": _STATE.get("vision_url"),
    })


@api.post("/score")
async def score_endpoint(
    file: UploadFile = File(...),
    bpm: int = Form(120),
    base_progression: Optional[str] = Form(default=None),
    frames_per_scene: int = Form(3),
    authorization: str = Header(default=""),
):
    _require_auth(authorization)

    # Lazy import the heavy modules so /health stays fast even on cold
    # restore (the snap=True hook already imports these for prod paths).
    from video_scoring.analyzer import AnalyzerConfig, VLMVideoAnalyzer
    from video_scoring.engine import FilmScoringEngine, ScoringSyncType
    from video_scoring.moondream_client import VisionClient

    # Disk-buffer the upload (PySceneDetect + OpenCV want a path).
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    tmp_root = tempfile.mkdtemp(prefix="vs_")
    vid_path = os.path.join(tmp_root, f"in_{uuid.uuid4().hex}{suffix}")
    with open(vid_path, "wb") as fh:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)

    size = os.path.getsize(vid_path)
    if size == 0:
        raise HTTPException(status_code=400, detail="empty upload")
    _log_json("video-score-start", filename=file.filename, bytes=size, bpm=bpm)

    api_key = _STATE["api_key"]
    vision_url = _STATE["vision_url"]
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def _emit(stage, payload=None):
        evt = (stage, payload or {})
        try:
            loop.call_soon_threadsafe(queue.put_nowait, evt)
        except RuntimeError:
            pass

    def _worker():
        start_t = time.monotonic()
        try:
            client = VisionClient(base_url=vision_url, api_key=api_key)
            cfg = AnalyzerConfig(frames_per_scene=max(1, min(8, frames_per_scene)))

            def progress(stage, payload):
                _emit(stage, payload)

            features = VLMVideoAnalyzer(
                vid_path, vision_client=client, config=cfg,
            ).analyze(progress=progress)

            _emit("midi", {})
            midi_path = os.path.join(tmp_root, "score.mid")
            FilmScoringEngine(bpm=bpm).generate_score(
                features=features,
                base_progression=_parse_progression(base_progression),
                scoring_approach=ScoringSyncType.TENSION_ARC,
                output_path=midi_path,
            )
            with open(midi_path, "rb") as fh:
                midi_b64 = base64.b64encode(fh.read()).decode("ascii")

            scene_data = []
            scene_changes = []
            duration = 0.0
            for f in features:
                d = {
                    "start_time": f.start_time,
                    "end_time": f.end_time,
                    "duration": f.duration,
                    "scene_id": f.scene_id,
                    "is_scene_start": f.is_scene_start,
                    "mood": f.mood.value,
                    "visual_tension": f.visual_tension,
                    "motion_intensity": f.motion_intensity,
                    "avg_brightness": f.avg_brightness,
                    "avg_saturation": f.avg_saturation,
                    "avg_hue": f.avg_hue,
                    "objects": list(f.objects),
                    "detected_text": list(f.detected_text),
                    "description": f.description,
                }
                scene_data.append(d)
                scene_changes.append(f.start_time)
                duration = max(duration, f.end_time)
            if scene_changes:
                scene_changes.append(duration)

            elapsed = time.monotonic() - start_t
            _log_json(
                "video-score-done",
                scenes=len(features),
                duration=duration,
                elapsed_s=round(elapsed, 2),
            )
            _emit("done", {
                "scene_data": scene_data,
                "scene_changes": scene_changes,
                "duration": duration,
                "midi_base64": midi_b64,
            })
        except Exception as exc:
            _log_json(
                "video-score-error",
                error=str(exc),
                elapsed_s=round(time.monotonic() - start_t, 2),
            )
            _emit("error", {"message": str(exc)[:512]})
        finally:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, ("__end__", {}))
            except RuntimeError:
                pass

    threading.Thread(target=_worker, daemon=True).start()

    async def _sse_stream():
        try:
            while True:
                stage, payload = await queue.get()
                if stage == "__end__":
                    break
                line = (
                    f"event: {stage}\n"
                    f"data: {json.dumps(payload, default=str)}\n\n"
                )
                yield line.encode("utf-8")
        finally:
            # Best-effort cleanup of the uploaded video temp dir.
            try:
                for root, dirs, files in os.walk(tmp_root, topdown=False):
                    for n in files:
                        try: os.remove(os.path.join(root, n))
                        except OSError: pass
                    for d in dirs:
                        try: os.rmdir(os.path.join(root, d))
                        except OSError: pass
                try: os.rmdir(tmp_root)
                except OSError: pass
            except Exception:
                pass

    return StreamingResponse(
        _sse_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Modal class — minimal wrapper that owns the GPU-state lifecycle and
# returns the module-level FastAPI app from its asgi_app hook.
# ---------------------------------------------------------------------------

@app.cls(
    image=image,
    cpu=4.0,
    memory=4096,
    secrets=[
        modal.Secret.from_name("doseedo-chatbot-gate"),  # provides VLLM_API_KEY
    ],
    timeout=600,
    scaledown_window=300,
    min_containers=0,
    max_containers=2,
    enable_memory_snapshot=True,
)
@modal.concurrent(max_inputs=4)
class VideoScoring:
    @modal.enter(snap=True)
    def setup_cpu_state(self):
        """Pre-snapshot init — pure-Python imports only.

        Touching cv2/numpy/scenedetect/fastapi here means the post-import
        Python heap gets memory-snapshotted and replayed on cold restore.
        Saves ~5-8 s vs. importing fresh on each cold start.
        """
        if "/root" not in sys.path:
            sys.path.insert(0, "/root")

        import cv2  # noqa: F401
        import mido  # noqa: F401
        import numpy  # noqa: F401
        import scenedetect  # noqa: F401

        # Eager import of the package modules so their byte-compiled
        # tree is in the snapshot.
        from video_scoring.engine import FilmScoringEngine  # noqa: F401
        from video_scoring.analyzer import VLMVideoAnalyzer  # noqa: F401
        from video_scoring.moondream_client import VisionClient  # noqa: F401

    @modal.enter(snap=False)
    def setup_post_snap(self):
        """Populate _STATE from secrets — env vars are NOT visible during
        snapshot capture, only after restore."""
        _STATE["api_key"] = os.environ["VLLM_API_KEY"]
        _STATE["vision_url"] = os.environ.get(
            "DOSEEDO_VISION_URL",
            "https://arlo--doseedo-chatbot-qwenchatbot-vision.modal.run",
        )

    @modal.asgi_app()
    def score(self):
        return api
