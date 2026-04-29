"""End-to-end: video file → MIDI score.

Glues the three pieces together:
    VLMVideoAnalyzer  (analyzer.py)
    FilmScoringEngine (engine.py)
    VisionClient      (moondream_client.py — to the Modal vision endpoint)

Designed so the analyzer's `VideoFeatures` output also matches what the legacy
ScoreAI `process_video` task returned at a higher level, so callers can either
get JSON scene metadata OR a generated MIDI file from a single run.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from typing import Dict, List, Optional

from .analyzer import AnalyzerConfig, VLMVideoAnalyzer
from .engine import FilmScoringEngine, ScoringSyncType, VideoFeatures
from .moondream_client import DEFAULT_VISION_URL, VisionClient


def score_video_to_midi(
    video_path: str,
    output_midi: Optional[str] = None,
    bpm: int = 120,
    base_progression: Optional[Dict[int, str]] = None,
    use_vision: bool = True,
    vision_url: Optional[str] = None,
    vision_api_key: Optional[str] = None,
    config: Optional[AnalyzerConfig] = None,
) -> Dict[str, object]:
    """Run the full pipeline. Returns {midi_path, scene_data, duration}.

    `use_vision=False` (or VLLM_API_KEY missing) skips Moondream and uses the
    HSV-only mood classifier — useful for offline tests.
    """
    client: Optional[VisionClient] = None
    if use_vision:
        try:
            client = VisionClient(
                base_url=vision_url or DEFAULT_VISION_URL,
                api_key=vision_api_key,
            )
        except RuntimeError as e:
            logging.warning("Vision disabled: %s", e)
            client = None

    analyzer = VLMVideoAnalyzer(video_path, vision_client=client, config=config)
    features: List[VideoFeatures] = analyzer.analyze()

    engine = FilmScoringEngine(bpm=bpm)
    midi_path = engine.generate_score(
        features=features,
        base_progression=base_progression,
        scoring_approach=ScoringSyncType.TENSION_ARC,
        output_path=output_midi,
    )

    return {
        "midi_path": midi_path,
        "duration": features[-1].end_time if features else 0.0,
        "scene_data": [_features_to_dict(f) for f in features],
    }


def _features_to_dict(f: VideoFeatures) -> dict:
    d = asdict(f)
    # Replace Enum with its string value for JSON serialisation.
    d["mood"] = f.mood.value
    return d


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_progression(spec: str) -> Dict[int, str]:
    """Parse 'Cm:0,Fm:4,G7:8,Cm:12' → {0:'Cm', 4:'Fm', 8:'G7', 12:'Cm'}."""
    out: Dict[int, str] = {}
    for item in spec.split(","):
        chord, beat = item.split(":")
        out[int(beat.strip())] = chord.strip()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="video_scoring", description="Video → MIDI film score")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--out", help="Output MIDI path (default: tempdir/film_score.mid)")
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument(
        "--base-progression",
        help="Optional base progression to morph, e.g. 'Cm:0,Fm:4,G7:8,Cm:12'",
    )
    parser.add_argument(
        "--no-vision", action="store_true",
        help="Skip Moondream — HSV-only mood classification.",
    )
    parser.add_argument("--vision-url", help="Override DOSEEDO_VISION_URL")
    parser.add_argument(
        "--frames-per-scene", type=int, default=3,
        help="Moondream calls per scene (default 3).",
    )
    parser.add_argument(
        "--report",
        help="Write scene-data JSON to this path (alongside MIDI).",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    base = _parse_progression(args.base_progression) if args.base_progression else None
    cfg = AnalyzerConfig(frames_per_scene=args.frames_per_scene)

    result = score_video_to_midi(
        video_path=args.video,
        output_midi=args.out,
        bpm=args.bpm,
        base_progression=base,
        use_vision=not args.no_vision,
        vision_url=args.vision_url,
        config=cfg,
    )

    print(f"midi:  {result['midi_path']}")
    print(f"scenes: {len(result['scene_data'])}, duration: {result['duration']:.2f}s")

    if args.report:
        with open(args.report, "w") as fh:
            json.dump(result, fh, indent=2, default=str)
        print(f"report: {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
