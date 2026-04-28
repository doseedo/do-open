# video_scoring

Video → MIDI film-scoring pipeline. Replaces the legacy ScoreAI stack
(`Do-Dev/home/arlo/ScoreAI/video_tasks.py`, GCV `videointelligence` + GCV
`vision`) with a Modal-hosted Moondream 2 vision model + local OpenCV /
PySceneDetect.

## Stack

| Step | Component | Notes |
|------|-----------|-------|
| Shot detection | PySceneDetect `ContentDetector` | Local, no API. Replaces GCV `SHOT_CHANGE_DETECTION`. |
| HSV mood / contrast | OpenCV | Local. Provides a ground-truth fallback when the VLM fails. |
| Motion | OpenCV Farneback optical flow | Local. Normalised to 0..1. |
| Scene VLM (mood, objects, on-screen text, action) | Moondream 2 via `modal/modal_chatbot.py` `/v1/vision/analyze` | Replaces GCV `LABEL_DETECTION`, `OBJECT_LOCALIZATION`, `IMAGE_PROPERTIES`, `TEXT_DETECTION`. |
| Score generation | `engine.FilmScoringEngine` | Ported from `Do-Dev/home/arlo/Data/film_scoring_engine.py`, with the empty-MIDI-output bug fixed. |

Whisper transcription is intentionally out of scope — the existing
`latent_whisper_student` stack handles that elsewhere.

## Setup

```bash
pip install -r video_scoring/requirements.txt
# ffmpeg / ffprobe must be on $PATH:
brew install ffmpeg     # macOS
# apt install ffmpeg    # Debian/Ubuntu

export VLLM_API_KEY="…"   # same secret used by the chatbot endpoint (modal secret: doseedo-chatbot-gate)
# Optional override; defaults to arlo--doseedo-chatbot-qwenchatbot-vision.modal.run:
# export DOSEEDO_VISION_URL="https://…modal.run"
```

## CLI

```bash
python -m video_scoring path/to/clip.mp4 \
  --out /tmp/score.mid \
  --bpm 110 \
  --base-progression "Cm:0,Fm:4,G7:8,Cm:12" \
  --frames-per-scene 3 \
  --report /tmp/scenes.json
```

`--no-vision` skips Moondream and produces an HSV-only score (useful for
offline tests; mood is always derived from colour).

## Programmatic

```python
from video_scoring import score_video_to_midi

result = score_video_to_midi(
    video_path="clip.mp4",
    output_midi="/tmp/score.mid",
    bpm=120,
)
# result["midi_path"], result["scene_data"], result["duration"]
```

If you want to run analysis without writing MIDI:

```python
from video_scoring.analyzer import VLMVideoAnalyzer
from video_scoring.moondream_client import VisionClient

features = VLMVideoAnalyzer("clip.mp4", vision_client=VisionClient()).analyze()
```

`features` is a list of `VideoFeatures` dataclasses, one per detected scene
(start_time, end_time, mood, visual_tension, motion_intensity, objects,
detected_text, description, …).

## Why Moondream 2 (not Qwen-VL)

`modal/modal_chatbot.py` co-locates Qwen3-14B-AWQ (text) with Moondream 2
(vision) on a single L4. Adding a third VLM (Qwen2.5-VL or Qwen3-VL) won't
fit in that container's VRAM budget — it would need its own Modal app.
Moondream's `query` is text-only output, so per-frame structured analysis is
done via a strict JSON-only prompt, parsed leniently in
`moondream_client.query_json`.

If Moondream's quality proves insufficient in practice, the upgrade path is:
1. Deploy Qwen2.5-VL-7B (or larger) as a *separate* Modal app, reusing the
   same `VLLM_API_KEY` secret.
2. Add a second `VisionClient` subclass that targets that endpoint.
3. Swap the client in `pipeline.score_video_to_midi`.

The analyzer/engine code does not change.

## Concurrency

Modal's `QwenChatbot` class declares `@modal.concurrent(max_inputs=16)`, so up
to 16 simultaneous vision calls share the single L4 container. The analyzer
uses a `ThreadPoolExecutor` (`AnalyzerConfig.max_workers`, default 4) which
sits well under that ceiling and leaves headroom for chat traffic.

## Decoupling vs the legacy code

* `engine.py` is **video-agnostic** — it has no `cv2` / `scenedetect` imports
  and is unit-testable with hand-built `VideoFeatures`.
* `analyzer.py` produces `VideoFeatures` and nothing else — swap it out for
  a non-Moondream analyzer without touching the engine.
* `moondream_client.py` is a thin HTTP wrapper that does not depend on
  anything else in the package — reusable from other call sites if you want
  ad-hoc vision queries.
