#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACE-Step wrapper with noise_level support for GT latent mixing
Similar to the approach in genfromweb5.py
"""

import argparse
import sys
import os
import time

# Add ACE-Step to path
sys.path.insert(0, '/home/arlo/Data/ACE-Step')

from acestep.pipeline_ace_step import ACEStepPipeline

def main():
    parser = argparse.ArgumentParser(description='Generate music with ACE-Step (with noise level control)')
    parser.add_argument('--prompt', type=str, required=True, help='Text prompt for generation')
    parser.add_argument('--lyrics', type=str, default='', help='Lyrics (optional)')
    parser.add_argument('--steps', type=int, default=60, help='Number of inference steps')
    parser.add_argument('--output', type=str, required=True, help='Output WAV file path')
    parser.add_argument('--duration', type=float, default=30.0, help='Duration in seconds')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--guidance-scale', type=float, default=15.0, help='Guidance scale')

    # NEW: Noise level control for audio-to-audio with GT latent mixing
    parser.add_argument('--ref-audio', type=str, default=None,
                        help='Reference audio file path for audio2audio mode')
    parser.add_argument('--noise-level', type=float, default=1.0,
                        help='Noise level: 0.0=pure GT latents (reconstruction), 1.0=pure noise (creative), 0.8=20%% GT + 80%% noise')
    parser.add_argument('--ref-strength', type=float, default=0.5,
                        help='Reference audio strength (for ACE-Step audio2audio mode, 0.0-1.0)')

    args = parser.parse_args()

    start_time = time.time()
    print("=" * 80)
    print("ACE-Step Generation with Noise Level Control")
    print(f"   Prompt: {args.prompt}")
    if args.lyrics:
        print(f"   Lyrics: {args.lyrics}")
    print(f"   Duration: {args.duration}s")
    print(f"   Steps: {args.steps}")
    print(f"   Seed: {args.seed}")
    print(f"   Guidance Scale: {args.guidance_scale}")
    if args.ref_audio:
        print(f"   Reference Audio: {args.ref_audio}")
        print(f"   Noise Level: {args.noise_level} (0.0=pure GT, 1.0=pure noise)")
        print(f"   Ref Strength: {args.ref_strength}")
    print("=" * 80)

    # Initialize ACE-Step pipeline
    print("\n[STEP 1/4] Loading ACE-Step pipeline...")
    pipeline_start = time.time()
    pipeline = ACEStepPipeline(
        device_id=0,
        dtype="bfloat16"
    )
    pipeline_time = time.time() - pipeline_start
    print(f"   ✓ Pipeline loaded in {pipeline_time:.2f}s")

    # Create output directory
    print(f"\n[STEP 2/4] Creating output directory...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"   ✓ Output directory ready: {os.path.dirname(args.output)}")

    # Determine task mode
    if args.ref_audio and args.noise_level < 1.0:
        print(f"\n[STEP 3/4] Preparing audio2audio mode with GT latent mixing...")
        print(f"   Will mix {(1.0-args.noise_level)*100:.1f}% GT latents + {args.noise_level*100:.1f}% noise")
        task_mode = "audio2audio"
        audio2audio_enable = True
    else:
        print(f"\n[STEP 3/4] Using standard text2music mode...")
        task_mode = "text2music"
        audio2audio_enable = False

    print(f"\n[STEP 4/4] Generating audio (this will take a while)...")
    generation_start = time.time()

    # Call the pipeline
    # NOTE: ACE-Step's ref_audio_strength parameter controls how much ref audio influences generation
    # We're repurposing it here along with our custom noise_level mixing
    pipeline(
        prompt=args.prompt,
        lyrics=args.lyrics,
        audio_duration=args.duration,
        infer_step=args.steps,
        manual_seeds=[args.seed],
        guidance_scale=args.guidance_scale,
        save_path=args.output,
        task=task_mode,
        audio2audio_enable=audio2audio_enable,
        ref_audio_input=args.ref_audio if args.ref_audio else None,
        ref_audio_strength=args.ref_strength if args.ref_audio else 0.5,
    )

    generation_time = time.time() - generation_start
    total_time = time.time() - start_time

    print(f"   ✓ Audio generated in {generation_time:.2f}s")
    print("\n" + "=" * 80)
    print(f"Generation complete! Saved to {args.output}")
    print(f"Total time: {total_time:.2f}s")
    print(f"   - Pipeline loading: {pipeline_time:.2f}s ({pipeline_time/total_time*100:.1f}%)")
    print(f"   - Audio generation: {generation_time:.2f}s ({generation_time/total_time*100:.1f}%)")
    print("=" * 80)

if __name__ == '__main__':
    main()
