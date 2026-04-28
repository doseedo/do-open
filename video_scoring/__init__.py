"""Video-to-MIDI film scoring pipeline.

Stack:
  PySceneDetect (shot boundaries)
  OpenCV       (HSV color, optical-flow motion)
  Moondream 2  (per-scene VLM mood/objects/text via Modal vision endpoint)
  mido         (MIDI emission from generated chord progressions)

Public surface:
  from video_scoring import score_video_to_midi
  from video_scoring.engine import (
      VideoFeatures, MoodCategory, TensionLevel,
      FilmScoringEngine, FilmScoringTechniques, LeitmotifEngine,
  )
  from video_scoring.analyzer import VLMVideoAnalyzer
  from video_scoring.moondream_client import VisionClient
"""

from .engine import (  # noqa: F401
    FilmScoringEngine,
    FilmScoringTechniques,
    LeitmotifEngine,
    MoodCategory,
    ScoringSyncType,
    TensionArc,
    TensionLevel,
    VideoFeatures,
)
from .pipeline import score_video_to_midi  # noqa: F401
