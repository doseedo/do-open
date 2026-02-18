#!/usr/bin/env python3
"""Inverse Synthesis — Patch Grabber CLI.

Takes an audio file, analyzes it, and outputs a Modulo synth patch JSON.
Handles real-world audio: loading, segmentation, pitch detection, optimization.

Usage:
    cd /home/arlo/do-repo/home/arlo/Modulo/inverse_patch

    # Single note audio → single patch
    python scripts/infer.py recording.wav

    # Multi-note recording → multiple patches
    python scripts/infer.py recording.wav --segment

    # Specify pitch manually
    python scripts/infer.py recording.wav --pitch 440

    # Save patch + re-synthesized audio
    python scripts/infer.py recording.wav -o output_dir/
"""

import sys
import os
import time
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(line_buffering=True)

from audio_input import load_audio, prepare_single, load_and_segment, normalize
from patch_schema import optimizer_result_to_patch, save_patch, patch_summary
from fast_dsp import (
    full_render, full_render_with_effects, fm_render,
    ALL_WF_FNS, SAMPLE_RATE, N_SAMPLES,
)


def render_patch_audio(patch):
    """Render audio from a patch dict using fast_dsp."""
    from patch_schema import patch_to_render_params
    params, wf_name, filter_type, pitch = patch_to_render_params(patch)

    if wf_name == 'fm':
        return fm_render(params, pitch)
    else:
        wf_fn = ALL_WF_FNS.get(wf_name, ALL_WF_FNS['saw'])
        if wf_name == 'noise':
            waveform = wf_fn()
        else:
            waveform = wf_fn(pitch)
        return full_render(params, waveform, filter_type=filter_type)


def save_wav(path, audio, sr=SAMPLE_RATE):
    """Save audio as WAV file."""
    import torchaudio
    import torch
    audio_t = torch.from_numpy(audio).float().unsqueeze(0)
    torchaudio.save(str(path), audio_t, sr)


def grab_patch(audio, pitch=None, verbose=True):
    """Core function: audio array → patch dict.

    Args:
        audio: numpy float32 array (mono, 44100Hz, ~2s)
        pitch: float Hz or None for auto-detect
        verbose: print optimization progress

    Returns:
        patch dict (see patch_schema.py)
    """
    from test_audio_domain import optimize_patch_full

    result = optimize_patch_full(audio, pitch=pitch, verbose=verbose)
    patch = optimizer_result_to_patch(result, pitch=pitch or result.get('pitch'))

    return patch, result


def grab_patch_from_file(path, pitch=None, verbose=True):
    """Load audio file → grab patch.

    Handles loading, resampling, trimming, normalization.
    Returns (patch, result, audio_2s).
    """
    audio, detected_pitch = prepare_single(path, pitch=pitch)
    use_pitch = pitch or detected_pitch

    if verbose:
        print(f"Input: {path}")
        print(f"Pitch: {use_pitch:.1f}Hz" if use_pitch else "Pitch: not detected")

    patch, result = grab_patch(audio, pitch=use_pitch, verbose=verbose)
    return patch, result, audio


def main():
    parser = argparse.ArgumentParser(
        description="Inverse Synthesis — grab synth patches from audio"
    )
    parser.add_argument("input", type=str, help="Path to input audio file")
    parser.add_argument("--pitch", "-p", type=float, default=None,
                        help="Override pitch in Hz (auto-detected if not given)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--segment", "-s", action="store_true",
                        help="Segment multi-note recording into individual notes")
    parser.add_argument("--max-notes", type=int, default=8,
                        help="Max notes to extract when segmenting (default: 8)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress optimization progress output")
    parser.add_argument("--render", "-r", action="store_true",
                        help="Also save re-synthesized audio from recovered patch")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: File not found: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet
    base_name = input_path.stem

    t_start = time.time()

    if args.segment:
        # Multi-note segmentation mode
        print(f"Segmenting {input_path}...")
        segments = load_and_segment(str(input_path), max_notes=args.max_notes)
        print(f"Found {len(segments)} note(s)")

        patches = []
        for i, seg in enumerate(segments):
            print(f"\n{'='*50}")
            print(f"Note {i+1}/{len(segments)} (start={seg['start_s']:.2f}s, "
                  f"dur={seg['duration_s']:.2f}s)")

            use_pitch = args.pitch or seg.get('pitch')
            if use_pitch and verbose:
                conf = seg.get('pitch_confidence', 0)
                print(f"  Pitch: {use_pitch:.1f}Hz (conf={conf:.2f})")
            elif verbose:
                print(f"  Pitch: not detected")

            patch, result = grab_patch(seg['audio'], pitch=use_pitch, verbose=verbose)
            patches.append(patch)

            # Save individual patch
            patch_path = output_dir / f"{base_name}_note{i+1}.json"
            save_patch(patch, patch_path)
            print(f"  Saved: {patch_path}")
            print(f"  {patch_summary(patch)}")

            if args.render:
                rendered = render_patch_audio(patch)
                render_path = output_dir / f"{base_name}_note{i+1}_synth.wav"
                save_wav(render_path, rendered)
                # Also save the target segment
                target_path = output_dir / f"{base_name}_note{i+1}_target.wav"
                save_wav(target_path, seg['audio'])

        elapsed = time.time() - t_start
        print(f"\n{'='*50}")
        print(f"Extracted {len(patches)} patch(es) in {elapsed:.1f}s")
        print(f"Output: {output_dir}")

    else:
        # Single-note mode
        patch, result, audio_2s = grab_patch_from_file(
            str(input_path), pitch=args.pitch, verbose=verbose
        )

        # Save patch
        patch_path = output_dir / f"{base_name}_patch.json"
        save_patch(patch, patch_path)

        elapsed = time.time() - t_start
        print(f"\n{'='*50}")
        print(f"Patch: {patch_summary(patch)}")
        print(f"Saved: {patch_path}")
        print(f"Time: {elapsed:.1f}s")

        if args.render:
            # Save re-synthesized audio
            rendered = render_patch_audio(patch)
            render_path = output_dir / f"{base_name}_synth.wav"
            save_wav(render_path, rendered)
            print(f"Synth: {render_path}")

            # Save target (normalized, 2s)
            target_path = output_dir / f"{base_name}_target.wav"
            save_wav(target_path, audio_2s)
            print(f"Target: {target_path}")


if __name__ == "__main__":
    main()
