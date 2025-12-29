#!/usr/bin/env python3
"""
Sliding Window Generation for Fast Musical Passages

This module provides a sliding window approach to handle fast/complex MIDI inputs
that the model struggles with at full tempo. Instead of using the tape speed hack
(slow down -> generate -> speed up) which introduces time-stretch artifacts, this
approach:

1. Splits conditioning into overlapping windows (e.g., 3 seconds, 50% overlap)
2. Generates each window independently (model sees manageable note density)
3. Crossfade blends the outputs with raised-cosine windows

This avoids time-stretch artifacts entirely since generation happens at natural tempo.

Author: Claude Code
Date: 2024-12-29
"""

import os
import sys
import time
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
import torchaudio
import torch.nn.functional as F


# ------------------------------------------------------------------------------
# Note Density Analysis
# ------------------------------------------------------------------------------

def compute_note_density(piano_roll: np.ndarray, window_frames: int = 43) -> np.ndarray:
    """
    Compute note density (notes per second) across the piano roll.

    Args:
        piano_roll: Piano roll array of shape [128, T] or [88, T]
        window_frames: Frames to average over (~1 second at 43fps)

    Returns:
        density: Array of shape [T] with notes-per-second at each frame
    """
    # Count active notes at each frame
    active_notes = (piano_roll > 0.1).sum(axis=0).astype(np.float32)  # [T]

    # Detect note onsets (new notes appearing)
    note_onsets = np.zeros_like(active_notes)
    note_onsets[1:] = np.maximum(0, np.diff((piano_roll > 0.1).astype(np.float32), axis=1).sum(axis=0))
    note_onsets[0] = active_notes[0]

    # Smooth with moving average to get density
    kernel = np.ones(window_frames) / window_frames
    if len(note_onsets) >= window_frames:
        density = np.convolve(note_onsets, kernel, mode='same')
    else:
        density = note_onsets

    # Convert to notes per second (43 fps)
    density = density * 43.066

    return density


def should_use_sliding_window(
    piano_roll: np.ndarray,
    density_threshold: float = 8.0,
    min_fast_region_frames: int = 43
) -> Tuple[bool, float, float]:
    """
    Determine if sliding window generation should be used based on note density.

    Args:
        piano_roll: Piano roll array [128, T] or [88, T]
        density_threshold: Notes per second threshold to trigger sliding window
        min_fast_region_frames: Minimum frames of fast content to trigger (~1 second)

    Returns:
        (should_use: bool, max_density: float, fast_region_percentage: float)
    """
    density = compute_note_density(piano_roll)
    max_density = float(density.max())
    fast_frames = (density > density_threshold).sum()
    fast_percentage = fast_frames / len(density) * 100

    should_use = fast_frames >= min_fast_region_frames

    return should_use, max_density, fast_percentage


# ------------------------------------------------------------------------------
# Conditioning Window Splitting
# ------------------------------------------------------------------------------

def split_conditioning_into_windows(
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    window_frames: int = 128,
    overlap_ratio: float = 0.5
) -> List[Dict]:
    """
    Split conditioning arrays into overlapping windows.

    Args:
        piano_roll: [128, T] or [88, T]
        amp, rframe, rbend: [T]
        window_frames: Frames per window (at 43fps, 128 frames ~ 3 seconds)
        overlap_ratio: Overlap between windows (0.5 = 50% overlap)

    Returns:
        List of dicts, each containing windowed conditioning:
        [{'piano_roll': [...], 'amp': [...], 'rframe': [...], 'rbend': [...],
          'start_frame': int, 'end_frame': int, 'actual_length': int}, ...]
    """
    T = piano_roll.shape[-1]
    hop_frames = int(window_frames * (1 - overlap_ratio))

    windows = []
    start = 0

    while start < T:
        end = min(start + window_frames, T)

        # Handle last window - extend back if too short
        if end - start < window_frames // 2 and len(windows) > 0:
            break

        actual_len = end - start

        window = {
            'piano_roll': piano_roll[:, start:end].copy(),
            'amp': amp[start:end].copy(),
            'rframe': rframe[start:end].copy(),
            'rbend': rbend[start:end].copy(),
            'start_frame': start,
            'end_frame': end,
            'actual_length': actual_len,
        }

        # Pad to full window size if needed
        if actual_len < window_frames:
            pad_len = window_frames - actual_len
            window['piano_roll'] = np.pad(window['piano_roll'], ((0, 0), (0, pad_len)), mode='constant')
            window['amp'] = np.pad(window['amp'], (0, pad_len), mode='constant')
            window['rframe'] = np.pad(window['rframe'], (0, pad_len), mode='constant')
            window['rbend'] = np.pad(window['rbend'], (0, pad_len), mode='constant')

        windows.append(window)
        start += hop_frames

    return windows


# ------------------------------------------------------------------------------
# Crossfade Blending
# ------------------------------------------------------------------------------

def create_crossfade_weights(length: int, fade_in: int, fade_out: int) -> np.ndarray:
    """
    Create crossfade weight array with raised cosine fades.

    Args:
        length: Total length of the window
        fade_in: Number of samples for fade in (0 for first window)
        fade_out: Number of samples for fade out (0 for last window)

    Returns:
        weights: Array of shape [length] with values 0-1
    """
    weights = np.ones(length)

    if fade_in > 0:
        # Raised cosine fade in: 0.5 * (1 - cos(pi * t))
        t = np.linspace(0, 1, fade_in)
        weights[:fade_in] = 0.5 * (1 - np.cos(np.pi * t))

    if fade_out > 0:
        # Raised cosine fade out: 0.5 * (1 + cos(pi * t))
        t = np.linspace(0, 1, fade_out)
        weights[-fade_out:] = 0.5 * (1 + np.cos(np.pi * t))

    return weights


def crossfade_blend_audio(
    audio_segments: List[torch.Tensor],
    window_info: List[Dict],
    sample_rate: int = 44100,
    fps: float = 43.066
) -> torch.Tensor:
    """
    Blend audio segments with crossfade based on window positions.

    Args:
        audio_segments: List of audio tensors [C, samples] or [samples]
        window_info: List of window dicts with 'start_frame', 'end_frame', 'actual_length'
        sample_rate: Audio sample rate
        fps: Conditioning frame rate

    Returns:
        blended: Single blended audio tensor [C, samples]
    """
    if len(audio_segments) == 1:
        seg = audio_segments[0]
        if seg.ndim == 1:
            seg = seg.unsqueeze(0)
        return seg

    # Calculate total length from the last window's end
    last_window = window_info[-1]
    total_frames = last_window['start_frame'] + last_window['actual_length']
    total_samples = int(total_frames / fps * sample_rate)

    # Detect number of channels from first segment
    first_seg = audio_segments[0]
    if first_seg.ndim == 1:
        num_channels = 1
    else:
        num_channels = first_seg.shape[0]

    # Initialize output and weight accumulator
    output = torch.zeros(num_channels, total_samples)
    weights = torch.zeros(total_samples)

    samples_per_frame = sample_rate / fps

    for i, (audio, winfo) in enumerate(zip(audio_segments, window_info)):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)

        start_sample = int(winfo['start_frame'] * samples_per_frame)
        actual_samples = int(winfo['actual_length'] * samples_per_frame)

        # Trim audio to actual length (remove padding)
        audio = audio[:, :actual_samples]

        end_sample = start_sample + audio.shape[-1]

        # Ensure we don't exceed output bounds
        if end_sample > total_samples:
            audio = audio[:, :total_samples - start_sample]
            end_sample = total_samples

        # Calculate fade lengths based on overlap
        if i > 0:
            prev_end = int((window_info[i-1]['start_frame'] + window_info[i-1]['actual_length']) * samples_per_frame)
            fade_in_samples = max(0, prev_end - start_sample)
        else:
            fade_in_samples = 0

        if i < len(audio_segments) - 1:
            next_start = int(window_info[i+1]['start_frame'] * samples_per_frame)
            fade_out_samples = max(0, end_sample - next_start)
        else:
            fade_out_samples = 0

        # Create weights for this segment
        seg_weights = create_crossfade_weights(
            audio.shape[-1],
            fade_in_samples,
            fade_out_samples
        )
        seg_weights = torch.from_numpy(seg_weights).float()

        # Add weighted audio to output
        seg_len = audio.shape[-1]
        output[:, start_sample:start_sample + seg_len] += audio * seg_weights
        weights[start_sample:start_sample + seg_len] += seg_weights

    # Normalize by weights (avoid division by zero)
    weights = weights.clamp(min=1e-8)
    output = output / weights.unsqueeze(0)

    return output


# ------------------------------------------------------------------------------
# Main Sliding Window Generation Function
# ------------------------------------------------------------------------------

def generate_sliding_window(
    model,
    generate_fn,
    piano_roll: np.ndarray,
    amp: np.ndarray,
    rframe: np.ndarray,
    rbend: np.ndarray,
    group: str,
    subgroup: str,
    window_seconds: float = 3.0,
    overlap_ratio: float = 0.5,
    fps: float = 43.066,
    audio_file: str = None,
    output_dir: str = "./generated_ui",
    **generate_kwargs
) -> str:
    """
    Generate audio using sliding window approach for fast passages.

    Splits conditioning into overlapping windows, generates each independently,
    then crossfade blends the outputs. This avoids the temporal resolution issues
    that occur when the model tries to handle very fast note sequences.

    Args:
        model: The generation model
        generate_fn: The generate() function to call for each window
        piano_roll: [128, T] piano roll conditioning
        amp, rframe, rbend: [T] conditioning arrays
        group, subgroup: Instrument identifiers
        window_seconds: Window duration in seconds (default 3.0)
        overlap_ratio: Window overlap (default 0.5 = 50%)
        fps: Conditioning frame rate
        audio_file: Path to source audio for GT latent extraction (optional)
        output_dir: Directory to save output
        **generate_kwargs: Additional args passed to generate()

    Returns:
        Path to generated audio file
    """
    window_frames = int(window_seconds * fps)
    T = piano_roll.shape[-1]
    total_duration = T / fps

    print(f"\n{'='*80}")
    print(f"SLIDING WINDOW GENERATION")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.2f}s ({T} frames)")
    print(f"Window size: {window_seconds:.1f}s ({window_frames} frames)")
    print(f"Overlap: {overlap_ratio*100:.0f}%")

    # Check note density
    should_window, max_density, fast_pct = should_use_sliding_window(piano_roll)
    print(f"Max note density: {max_density:.1f} notes/sec")
    print(f"Fast regions: {fast_pct:.1f}% of content")

    # Load source audio for GT latent extraction if provided
    source_audio = None
    source_sr = 44100
    if audio_file and os.path.exists(audio_file):
        print(f"Loading source audio for GT latents: {Path(audio_file).name}")
        source_audio, source_sr = torchaudio.load(audio_file)
        if source_audio.shape[0] > 1:
            source_audio = source_audio.mean(dim=0, keepdim=True)
        print(f"   Source audio: {source_audio.shape[-1]} samples ({source_audio.shape[-1]/source_sr:.2f}s)")

    # Split into windows
    windows = split_conditioning_into_windows(
        piano_roll, amp, rframe, rbend,
        window_frames=window_frames,
        overlap_ratio=overlap_ratio
    )
    print(f"Split into {len(windows)} windows")

    # Create empty encodec for each window
    encodec_length = window_frames // 4
    empty_encodec = torch.zeros((1, 8, encodec_length), dtype=torch.long)

    # Create temp directory for audio slices
    temp_dir = tempfile.mkdtemp(prefix="sliding_window_")

    # Generate each window
    audio_segments = []
    seed = generate_kwargs.get('seed', -1)
    if seed <= 0:
        seed = torch.seed() % 2**31

    for i, window in enumerate(windows):
        print(f"\n   Window {i+1}/{len(windows)}: frames {window['start_frame']}-{window['end_frame']}")

        window_seed = (seed + i * 1000) % 2**31
        window_duration = window['actual_length'] / fps
        original_audio_length = int(window_duration * 44100)

        # Slice source audio for this window if available
        window_audio_file = None
        if source_audio is not None:
            start_sample = int(window['start_frame'] / fps * source_sr)
            end_sample = int((window['start_frame'] + window['actual_length']) / fps * source_sr)
            end_sample = min(end_sample, source_audio.shape[-1])

            if end_sample > start_sample:
                window_audio = source_audio[:, start_sample:end_sample]
                window_audio_file = os.path.join(temp_dir, f"window_{i}.wav")
                torchaudio.save(window_audio_file, window_audio, source_sr)
                print(f"      Sliced GT audio: {start_sample}-{end_sample} samples ({window_audio.shape[-1]/source_sr:.2f}s)")

        # Generate this window
        window_kwargs = generate_kwargs.copy()
        window_kwargs['seed'] = window_seed
        window_kwargs['original_audio_length'] = original_audio_length
        window_kwargs['target_audio_duration'] = window_duration
        if window_audio_file:
            window_kwargs['audio_file'] = window_audio_file

        output_path = generate_fn(
            model=model,
            piano_roll=window['piano_roll'],
            amp=window['amp'],
            rframe=window['rframe'],
            rbend=window['rbend'],
            encodec_tokens=empty_encodec,
            group=group,
            subgroup=subgroup,
            **window_kwargs
        )

        # Load the generated audio
        audio, sr = torchaudio.load(output_path)

        # Trim to actual length
        expected_samples = int(window['actual_length'] / fps * sr)
        if audio.shape[-1] > expected_samples:
            audio = audio[:, :expected_samples]

        audio_segments.append(audio)
        print(f"   Window {i+1} generated: {audio.shape[-1]} samples")

    # Crossfade blend all segments
    print(f"\nBlending {len(audio_segments)} segments with crossfade...")
    blended = crossfade_blend_audio(audio_segments, windows, sample_rate=44100, fps=fps)

    # Save blended output
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_sliding_window_seed{seed}.wav"

    torchaudio.save(str(out_path), blended, 44100)

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"\nSliding window generation complete: {out_path}")
    print(f"   Duration: {blended.shape[-1] / 44100:.2f}s")
    print(f"{'='*80}\n")

    return str(out_path)


# ------------------------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("Sliding Window Generation Module")
    print("=" * 50)
    print()
    print("This module provides functions for sliding window generation:")
    print()
    print("  - compute_note_density(piano_roll)")
    print("    Compute notes-per-second across the piano roll")
    print()
    print("  - should_use_sliding_window(piano_roll)")
    print("    Check if sliding window is recommended based on note density")
    print()
    print("  - split_conditioning_into_windows(piano_roll, amp, rframe, rbend)")
    print("    Split conditioning arrays into overlapping windows")
    print()
    print("  - crossfade_blend_audio(audio_segments, window_info)")
    print("    Blend audio segments with raised-cosine crossfade")
    print()
    print("  - generate_sliding_window(model, generate_fn, ...)")
    print("    Main function to generate with sliding window approach")
    print()
    print("See sliding_window_generation.md for full documentation.")
