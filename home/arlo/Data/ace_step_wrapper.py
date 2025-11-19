#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE-Step wrapper for simple text-to-music generation
Uses the official ACE-Step pipeline
"""

import argparse
import sys
import os
import time

# Add ACE-Step to path
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from acestep.pipeline_ace_step import ACEStepPipeline

def main():
    parser = argparse.ArgumentParser(description='Generate music with ACE-Step')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--lyrics', type=str, default='', help='Lyrics (optional)')
    parser.add_argument('--steps', type=int, default=60, help='Number of inference steps')
    parser.add_argument('--output', type=str, required=True, help='Output WAV file path')
    parser.add_argument('--duration', type=float, default=30.0, help='Duration in seconds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--guidance-scale', type=float, default=15.0, help='Guidance scale')

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 60)
    print("ACE-Step Generation Starting...")
    print(f"   Prompt: {args.prompt}")
    print(f"   Duration: {args.duration}s")
    print(f"   Steps: {args.steps}")
    print(f"   Seed: {args.seed}")
    print(f"   Guidance Scale: {args.guidance_scale}")
    print("=" * 60)

    # Initialize ACE-Step pipeline
    print("\n[STEP 1/3] Loading ACE-Step pipeline...")
    pipeline_start = time.time()
    pipeline = ACEStepPipeline(
        device_id=0,
        dtype="bfloat16"
    )
    pipeline_time = time.time() - pipeline_start
    print(f"   ✓ Pipeline loaded in {pipeline_time:.2f}s")

    # Generate audio - ACE-Step saves directly to save_path
    print(f"\n[STEP 2/3] Creating output directory...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"   ✓ Output directory ready: {os.path.dirname(args.output)}")

    print(f"\n[STEP 3/3] Generating audio (this will take a while)...")
    generation_start = time.time()

    pipeline(
        prompt=args.prompt,
        lyrics=args.lyrics,
        audio_duration=args.duration,
        infer_step=args.steps,
        manual_seeds=[args.seed],
        guidance_scale=args.guidance_scale,
        save_path=args.output
    )

    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    print(f"   ✓ Audio generated in {generation_time:.2f}s")
    print("\n" + "=" * 60)
    print(f"Generation complete! Saved to {args.output}")
    print(f"Total time: {total_time:.2f}s")
    print(f"   - Pipeline loading: {pipeline_time:.2f}s ({pipeline_time/total_time*100:.1f}%)")
    print(f"   - Audio generation: {generation_time:.2f}s ({generation_time/total_time*100:.1f}%)")
    print("=" * 60)

if __name__ == '__main__':
    main()
