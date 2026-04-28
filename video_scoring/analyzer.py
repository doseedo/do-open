"""Video → list[VideoFeatures].

Replaces ScoreAI's GCV `videointelligence` + GCV `vision` per-frame analysis
with a local + Modal-Moondream stack:

    PySceneDetect  → shot boundaries          (was GCV SHOT_CHANGE_DETECTION)
    OpenCV HSV     → brightness/saturation/hue per scene
    OpenCV Farneback optical flow → motion_intensity
    Moondream 2 (Modal) → mood + dominant colours + objects + on-screen text
                                              (was GCV LABEL/OBJECT/TEXT/IMAGE_PROPERTIES)

Whisper transcription is intentionally NOT done here — keep that pipeline
separate (matches the existing latent_whisper_student stack).

Frame extraction is in-process via OpenCV (`VideoCapture.read` +
`cv2.imencode`). The previous version shelled out to ffmpeg per frame,
which (a) added ~50–100 ms of fork/exec per frame, (b) required ffmpeg on
PATH at runtime, and (c) wrote N temp files for no good reason. The
in-process path keeps frame bytes in memory and hands them straight to the
VisionClient.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import subprocess
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from .engine import MoodCategory, VideoFeatures
from .moondream_client import VisionClient

log = logging.getLogger(__name__)


_MOOD_BY_NAME = {m.value: m for m in MoodCategory}


_VLM_PROMPT = (
    "Look at this film frame. Reply with a single JSON object, no prose, "
    "with these keys:\n"
    '  "mood": one of '
    "[warm_bright, warm_dark, cool_bright, cool_dark, saturated, "
    "desaturated, high_contrast, low_contrast]\n"
    '  "tension": float 0.0-1.0 (emotional intensity)\n'
    '  "objects": list of strings (visible subjects, max 6)\n'
    '  "text": list of strings (any on-screen text/captions)\n'
    '  "description": one short sentence of the action.'
)

# Progress callback type. Called with (stage, payload). Stages emitted:
#   ("shots", {"count": N})
#   ("scene", {"i": k, "of": N, "start": float, "end": float})
#   ("scene_done", {"i": k, "of": N, "mood": str, "tension": float})
ProgressCB = Callable[[str, dict], None]


@dataclass
class AnalyzerConfig:
    framerate: float = 24.0
    scene_threshold: float = 27.0          # PySceneDetect ContentDetector
    min_scene_duration: float = 1.5        # drop scenes shorter than this
    frames_per_scene: int = 3              # Moondream calls per scene
    max_workers: int = 8                   # bounded by Modal max_inputs=16; chatbot uses other half
    motion_max_flow: float = 5.0           # for normalising optical flow → 0..1
    jpeg_quality: int = 85                 # cv2.imencode quality for VLM frames


class VLMVideoAnalyzer:
    """End-to-end producer of `VideoFeatures` for one video file."""

    def __init__(
        self,
        video_path: str,
        vision_client: Optional[VisionClient] = None,
        config: Optional[AnalyzerConfig] = None,
    ) -> None:
        if not os.path.exists(video_path):
            raise FileNotFoundError(video_path)
        self.video_path = video_path
        self.config = config or AnalyzerConfig()
        # vision_client may be None — analyzer falls back to OpenCV-only mood
        self.vision = vision_client

    # --- public -----------------------------------------------------------

    def analyze(self, progress: Optional[ProgressCB] = None) -> List[VideoFeatures]:
        scenes = self._detect_scenes()
        n = len(scenes)
        if progress:
            progress("shots", {"count": n})
        log.info("video=%s scenes=%d", os.path.basename(self.video_path), n)

        out: List[VideoFeatures] = []
        for idx, (start, end) in enumerate(scenes):
            if progress:
                progress("scene", {"i": idx, "of": n, "start": start, "end": end})
            feat = VideoFeatures(
                start_time=start,
                end_time=end,
                duration=end - start,
                is_scene_start=True,
                scene_id=idx,
            )
            self._fill_color_and_motion(feat)
            if self.vision is not None:
                self._fill_vlm(feat)
            else:
                feat.mood = _classify_mood_hsv(
                    feat.avg_brightness, feat.avg_saturation, feat.avg_hue
                )
                feat.visual_tension = _hsv_tension(feat)
            out.append(feat)
            if progress:
                progress("scene_done", {
                    "i": idx, "of": n,
                    "mood": feat.mood.value,
                    "tension": feat.visual_tension,
                })
        return out

    # --- shots ------------------------------------------------------------

    def _detect_scenes(self) -> List[Tuple[float, float]]:
        try:
            from scenedetect import open_video, SceneManager
            from scenedetect.detectors import ContentDetector
        except ImportError:
            log.warning("scenedetect unavailable, treating video as one scene")
            return [(0.0, _video_duration(self.video_path))]

        try:
            video = open_video(self.video_path)
            manager = SceneManager()
            manager.add_detector(ContentDetector(threshold=self.config.scene_threshold))
            manager.detect_scenes(video=video)
            scene_list = manager.get_scene_list()
            if not scene_list:
                return [(0.0, _video_duration(self.video_path))]
            min_dur = self.config.min_scene_duration
            scenes = [
                (s[0].get_seconds(), s[1].get_seconds())
                for s in scene_list
                if (s[1].get_seconds() - s[0].get_seconds()) >= min_dur
            ]
            return scenes if scenes else [(0.0, _video_duration(self.video_path))]
        except Exception as e:
            log.warning("scene detection failed: %s — falling back to single-scene", e)
            return [(0.0, _video_duration(self.video_path))]

    # --- HSV + optical flow ----------------------------------------------

    def _fill_color_and_motion(self, feat: VideoFeatures) -> None:
        cap = cv2.VideoCapture(self.video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.framerate
            start_f = int(feat.start_time * fps)
            end_f = int(feat.end_time * fps)
            n_samples = max(2, min(8, (end_f - start_f) // max(int(fps), 1)))
            if n_samples < 2:
                return
            indices = np.linspace(start_f, end_f - 1, n_samples, dtype=int)

            brightness, saturation, hue = [], [], []
            prev_gray = None
            flow_mags: List[float] = []

            for fi in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv)
                brightness.append(float(np.mean(v) / 255.0))
                saturation.append(float(np.mean(s) / 255.0))
                hue.append(float(np.mean(h)) * 2.0)  # OpenCV H is 0..179

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if prev_gray is not None and prev_gray.shape == gray.shape:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0,
                    )
                    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                    flow_mags.append(float(np.mean(mag)))
                prev_gray = gray

            if brightness:
                feat.avg_brightness = float(np.mean(brightness))
                feat.avg_saturation = float(np.mean(saturation))
                feat.avg_hue = float(np.mean(hue))
                feat.contrast_level = float(np.std(brightness))
            if flow_mags:
                norm = max(self.config.motion_max_flow, 1e-6)
                feat.motion_intensity = float(min(1.0, np.mean(flow_mags) / norm))
        finally:
            cap.release()

    # --- VLM --------------------------------------------------------------

    def _fill_vlm(self, feat: VideoFeatures) -> None:
        """Sample N frames in-memory, query Moondream in parallel, merge results."""
        cfg = self.config
        frame_blobs = self._extract_frame_jpegs(feat.start_time, feat.end_time, cfg.frames_per_scene)
        if not frame_blobs:
            feat.mood = _classify_mood_hsv(
                feat.avg_brightness, feat.avg_saturation, feat.avg_hue
            )
            feat.visual_tension = _hsv_tension(feat)
            return

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=cfg.max_workers) as ex:
            futs = [ex.submit(self._query_one_bytes, b) for b in frame_blobs]
            for f in concurrent.futures.as_completed(futs):
                try:
                    results.append(f.result())
                except Exception as e:
                    log.warning("Moondream query failed for scene %d: %s", feat.scene_id, e)

        # Aggregate
        moods = [r.get("mood") for r in results if isinstance(r.get("mood"), str)]
        tensions = [_clamp01(r.get("tension")) for r in results if r.get("tension") is not None]
        objects: List[str] = []
        texts: List[str] = []
        descs: List[str] = []
        for r in results:
            for o in (r.get("objects") or [])[:6]:
                if isinstance(o, str) and o not in objects:
                    objects.append(o)
            for t in (r.get("text") or [])[:6]:
                if isinstance(t, str) and t not in texts:
                    texts.append(t)
            d = r.get("description")
            if isinstance(d, str) and d:
                descs.append(d)

        # Pick most-voted mood, fall back to HSV classifier.
        if moods:
            from collections import Counter
            valid = [m for m in moods if m in _MOOD_BY_NAME]
            if valid:
                top, _ = Counter(valid).most_common(1)[0]
                feat.mood = _MOOD_BY_NAME.get(top, _classify_mood_hsv(
                    feat.avg_brightness, feat.avg_saturation, feat.avg_hue,
                ))
            else:
                feat.mood = _classify_mood_hsv(
                    feat.avg_brightness, feat.avg_saturation, feat.avg_hue,
                )
        else:
            feat.mood = _classify_mood_hsv(
                feat.avg_brightness, feat.avg_saturation, feat.avg_hue,
            )

        # Tension: blend VLM mean with HSV-derived + motion to keep it grounded
        hsv_tension = _hsv_tension(feat)
        vlm_tension = float(np.mean(tensions)) if tensions else hsv_tension
        feat.visual_tension = float(np.clip(
            0.5 * vlm_tension + 0.3 * hsv_tension + 0.2 * feat.motion_intensity,
            0.0, 1.0,
        ))
        feat.objects = objects
        feat.detected_text = texts
        feat.description = " ".join(descs)[:512]

    def _query_one_bytes(self, jpeg: bytes) -> dict:
        return self.vision.query_json(jpeg, _VLM_PROMPT)  # type: ignore[union-attr]

    def _extract_frame_jpegs(self, start: float, end: float, n: int) -> List[bytes]:
        """OpenCV-only: seek to N evenly spaced timestamps in [start, end),
        decode the frame, encode JPEG bytes in memory.

        Replaces the previous ffmpeg-subprocess-per-frame implementation.
        """
        if n <= 0 or end <= start:
            return []
        cap = cv2.VideoCapture(self.video_path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or self.config.framerate
            timestamps = np.linspace(start, end, n + 1)[:-1] + (end - start) / (2 * n)
            quality = max(50, min(95, self.config.jpeg_quality))
            out: List[bytes] = []
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, float(t) * 1000.0)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                if ok2:
                    out.append(buf.tobytes())
            return out
        finally:
            cap.release()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _classify_mood_hsv(brightness: float, saturation: float, hue: float) -> MoodCategory:
    is_warm = (0 <= hue <= 60) or (300 <= hue <= 360)
    is_bright = brightness > 0.5
    is_saturated = saturation > 0.4
    if is_warm and is_bright:
        return MoodCategory.WARM_BRIGHT
    if is_warm and not is_bright:
        return MoodCategory.WARM_DARK
    if not is_warm and is_bright:
        return MoodCategory.COOL_BRIGHT
    if not is_warm and not is_bright:
        return MoodCategory.COOL_DARK
    return MoodCategory.SATURATED if is_saturated else MoodCategory.DESATURATED


def _hsv_tension(feat: VideoFeatures) -> float:
    return float(np.clip(
        (1.0 - feat.avg_brightness) * 0.5 + feat.avg_saturation * 0.5,
        0.0, 1.0,
    ))


def _video_duration(path: str) -> float:
    """Try OpenCV first (no subprocess), fall back to ffprobe if installed."""
    try:
        cap = cv2.VideoCapture(path)
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            if fps and n_frames:
                return float(n_frames / fps)
        finally:
            cap.release()
    except Exception:
        pass
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1", path,
            ],
            capture_output=True, check=True, text=True,
        )
        return float(out.stdout.strip())
    except (subprocess.CalledProcessError, ValueError, FileNotFoundError):
        return 0.0


def _clamp01(v) -> float:
    try:
        return float(max(0.0, min(1.0, float(v))))
    except (TypeError, ValueError):
        return 0.0
