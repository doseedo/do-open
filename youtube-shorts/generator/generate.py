#!/usr/bin/env python3
"""Generate physics animation videos for YouTube Shorts.

Usage:
    python generate.py --scene plinko --count 1
    python generate.py --scene marble_race --count 3
    python generate.py --scene random --count 5
"""

import argparse
import json
import os
import random
import sys
import time

# Ensure parent directory is on path so 'generator' package resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set headless before importing pygame
os.environ["SDL_VIDEODRIVER"] = "dummy"

import pygame
pygame.init()
pygame.display.set_mode((1, 1))

from generator import config
from generator.renderer import render_video
from generator.scenes import plinko, marble_race

SCENES = {
    "plinko": plinko,
    "marble_race": marble_race,
}


def generate_video(scene_name: str, video_num: int):
    """Generate a single video + sidecar JSON."""
    scene_module = SCENES[scene_name]

    timestamp = int(time.time())
    filename = f"{scene_name}_{timestamp}_{video_num}"
    video_path = os.path.join(config.OUTPUT_DIR, f"{filename}.mp4")
    json_path = os.path.join(config.OUTPUT_DIR, f"{filename}.json")

    seed = random.randint(0, 2**32 - 1)
    print(f"Generating {scene_name} #{video_num} (seed={seed})...")

    frame_gen = scene_module.run(seed=seed)
    render_video(video_path, frame_gen)

    metadata = scene_module.get_metadata(video_num)
    with open(json_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Video: {video_path}")
    print(f"  Metadata: {json_path}")
    return video_path, json_path


def main():
    parser = argparse.ArgumentParser(description="Generate physics animation Shorts")
    parser.add_argument(
        "--scene",
        choices=["plinko", "marble_race", "random"],
        default="random",
        help="Scene type to generate",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of videos to generate",
    )
    args = parser.parse_args()

    for i in range(args.count):
        if args.scene == "random":
            scene_name = random.choice(list(SCENES.keys()))
        else:
            scene_name = args.scene

        video_num = random.randint(1, 9999)
        generate_video(scene_name, video_num)

    print(f"\nDone! Generated {args.count} video(s) in {config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
