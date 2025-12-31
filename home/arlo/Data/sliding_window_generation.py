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

    # Clamp fade lengths to not exceed available space
    fade_in = min(fade_in, length)
    fade_out = min(fade_out, length - fade_in)  # Don't overlap with fade_in

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
# Latent-Space Blending
# ------------------------------------------------------------------------------

def crossfade_blend_latents(
    latent_segments: List[torch.Tensor],
    window_info: List[Dict],
    latent_fps: float = 10.766  # ~43.066 / 4 for latent space
) -> torch.Tensor:
    """
    Blend latent segments with crossfade BEFORE decoding.

    This produces more coherent results than audio-space crossfade because
    timbral characteristics are better preserved when blending in latent space.

    Args:
        latent_segments: List of latent tensors [1, 8, 16, T] or [8, 16, T]
        window_info: List of window dicts with 'start_frame', 'end_frame', 'actual_length'
        latent_fps: Latent frame rate (conditioning_fps / 4)

    Returns:
        blended: Single blended latent tensor [1, 8, 16, T_total]
    """
    if len(latent_segments) == 1:
        lat = latent_segments[0]
        if lat.ndim == 3:
            lat = lat.unsqueeze(0)
        return lat

    # Get latent dimensions from first segment
    first_lat = latent_segments[0]
    if first_lat.ndim == 3:
        first_lat = first_lat.unsqueeze(0)
    batch, codebooks, channels, T_first = first_lat.shape  # [1, 8, 16, T]

    # Calculate total length from the last window's end
    # Note: Latent temporal dimension matches conditioning (NOT 4x downsampled)
    last_window = window_info[-1]
    total_cond_frames = last_window['start_frame'] + last_window['actual_length']
    total_latent_frames = total_cond_frames  # Latent matches conditioning length

    # Initialize output and weight accumulator
    device = first_lat.device
    dtype = first_lat.dtype
    output = torch.zeros(batch, codebooks, channels, total_latent_frames, device=device, dtype=dtype)
    weights = torch.zeros(total_latent_frames, device=device, dtype=dtype)

    conditioning_fps = 43.066

    for i, (latent, winfo) in enumerate(zip(latent_segments, window_info)):
        if latent.ndim == 3:
            latent = latent.unsqueeze(0)

        # Latent frames match conditioning frames (no downsampling)
        start_latent = winfo['start_frame']
        actual_latent_len = winfo['actual_length']

        # Trim latent to actual length
        latent = latent[..., :actual_latent_len]

        end_latent = start_latent + latent.shape[-1]

        # Ensure we don't exceed output bounds
        if end_latent > total_latent_frames:
            latent = latent[..., :total_latent_frames - start_latent]
            end_latent = total_latent_frames

        # Calculate fade lengths based on overlap (no /4 - latent matches conditioning)
        if i > 0:
            prev_end = window_info[i-1]['start_frame'] + window_info[i-1]['actual_length']
            fade_in_frames = max(0, prev_end - start_latent)
        else:
            fade_in_frames = 0

        if i < len(latent_segments) - 1:
            next_start = window_info[i+1]['start_frame']
            fade_out_frames = max(0, end_latent - next_start)
        else:
            fade_out_frames = 0

        # Create weights for this segment
        seg_weights = create_crossfade_weights(
            latent.shape[-1],
            fade_in_frames,
            fade_out_frames
        )
        seg_weights = torch.from_numpy(seg_weights).to(device=device, dtype=dtype)

        # Add weighted latent to output (broadcast weights across all dimensions)
        seg_len = latent.shape[-1]
        output[..., start_latent:start_latent + seg_len] += latent * seg_weights
        weights[start_latent:start_latent + seg_len] += seg_weights

    # Normalize by weights (avoid division by zero)
    weights = weights.clamp(min=1e-8)
    output = output / weights.view(1, 1, 1, -1)

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
        **generate_kwargs: Additional args passed to generate(), including:
            - noise_level: 0.0-1.0, how much noise vs GT latent (default 1.0)
            - post_gen_gt_mix: 0.0-1.0, mix GT AFTER generation at native resolution
              RECOMMENDED: Use noise_level=1.0 + post_gen_gt_mix=0.1 instead of
              noise_level=0.9, because pre-gen mixing requires 4x interpolation
              which destroys timbre. Post-gen mixing preserves GT timbre.
            - cfg_weight, steps, t0, adapter_scale, inst_boost, etc.

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

        # Use SAME seed for all windows to maintain consistent timbre
        window_seed = seed  # Critical: don't vary seed between windows
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
# Latent-Space Blending Sliding Window Generation
# ------------------------------------------------------------------------------

def generate_sliding_window_latent_blend(
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
    sr_out: int = 44100,
    use_overlap_decoder: bool = True,
    **generate_kwargs
) -> str:
    """
    Generate audio using sliding window with LATENT-SPACE blending.

    This is superior to audio-space crossfade because blending happens before
    decoding, preserving timbral coherence across windows.

    Key differences from generate_sliding_window:
    1. Generates latents instead of audio for each window
    2. Blends latents with crossfade
    3. Decodes the blended latent ONCE at the end

    Args:
        model: The generation model (with model.dcae for decoding)
        generate_fn: The generate() function (must support return_latents=True)
        piano_roll: [128, T] piano roll conditioning
        amp, rframe, rbend: [T] conditioning arrays
        group, subgroup: Instrument identifiers
        window_seconds: Window duration in seconds (default 3.0)
        overlap_ratio: Window overlap (default 0.5 = 50%)
        fps: Conditioning frame rate
        audio_file: Path to source audio for GT latent extraction (optional)
        output_dir: Directory to save output
        sr_out: Output sample rate
        use_overlap_decoder: Whether to use overlap decoder
        **generate_kwargs: Additional args passed to generate()

    Returns:
        Path to generated audio file
    """
    window_frames = int(window_seconds * fps)
    T = piano_roll.shape[-1]
    total_duration = T / fps

    print(f"\n{'='*80}")
    print(f"SLIDING WINDOW GENERATION (LATENT-SPACE BLENDING)")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.2f}s ({T} frames)")
    print(f"Window size: {window_seconds:.1f}s ({window_frames} frames)")
    print(f"Overlap: {overlap_ratio*100:.0f}%")

    # Check note density
    should_window, max_density, fast_pct = should_use_sliding_window(piano_roll)
    print(f"Max note density: {max_density:.1f} notes/sec")
    print(f"Fast regions: {fast_pct:.1f}% of content")

    # Debug: Check where notes exist in original piano roll
    pr_activity = (piano_roll > 0.1).sum(axis=0)  # [T] - activity per frame
    active_frames = np.where(pr_activity > 0)[0]
    if len(active_frames) > 0:
        first_active = active_frames[0]
        last_active = active_frames[-1]
        print(f"🔍 Original piano roll: notes in frames {first_active}-{last_active} ({last_active/fps:.2f}s)")
        print(f"   Total active frames: {len(active_frames)}/{T} ({100*len(active_frames)/T:.1f}%)")
    else:
        print(f"⚠️ WARNING: Original piano roll has NO active notes!")

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
    temp_dir = tempfile.mkdtemp(prefix="sliding_window_latent_")

    # Generate latents for each window
    latent_segments = []
    seed = generate_kwargs.get('seed', -1)
    if seed <= 0:
        seed = torch.seed() % 2**31

    for i, window in enumerate(windows):
        print(f"\n   Window {i+1}/{len(windows)}: frames {window['start_frame']}-{window['end_frame']}")

        # Debug: show piano roll activity for this window
        pr_active = np.sum(window['piano_roll'] > 0.1)
        pr_max = np.max(window['piano_roll'])
        amp_sum = np.sum(window['amp'])
        print(f"      🎹 Piano roll: {pr_active} active cells, max={pr_max:.3f}, amp_sum={amp_sum:.3f}")

        # Use SAME seed for all windows to maintain consistent timbre
        window_seed = seed  # Critical: don't vary seed between windows
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

        # Generate this window - GET LATENT instead of audio
        window_kwargs = generate_kwargs.copy()
        window_kwargs['seed'] = window_seed
        window_kwargs['original_audio_length'] = original_audio_length
        window_kwargs['target_audio_duration'] = window_duration
        window_kwargs['return_latents'] = True  # KEY: get latent, not audio
        window_kwargs['sr_out'] = sr_out  # Required parameter
        if window_audio_file:
            window_kwargs['audio_file'] = window_audio_file

        latent = generate_fn(
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

        # Trim latent to actual length (latent matches conditioning, no downsampling)
        expected_latent_frames = window['actual_length']
        if latent.shape[-1] > expected_latent_frames:
            latent = latent[..., :expected_latent_frames]

        latent_segments.append(latent)
        print(f"   Window {i+1} latent: {latent.shape}")

    # Blend latents in latent space
    print(f"\n🔧 Blending {len(latent_segments)} latents in LATENT SPACE...")
    blended_latent = crossfade_blend_latents(latent_segments, windows)
    print(f"   Blended latent shape: {blended_latent.shape}")

    # CRITICAL FIX: Blended latent is at conditioning fps (43.066), but DCAE expects
    # latent at its native resolution (~10.77 fps = 44100/4096). Downsample before decode.
    DCAE_HOP = 4096
    DCAE_SR = 44100
    DCAE_FPS = DCAE_SR / DCAE_HOP  # ~10.766 fps

    total_cond_frames = windows[-1]['start_frame'] + windows[-1]['actual_length']
    total_duration_seconds = total_cond_frames / fps
    target_latent_frames = int(round(total_duration_seconds * DCAE_FPS))

    if blended_latent.shape[-1] != target_latent_frames:
        print(f"   🔄 Resampling latent for DCAE: {blended_latent.shape[-1]} → {target_latent_frames} frames")
        print(f"      (conditioning at {fps:.1f}fps → DCAE at {DCAE_FPS:.2f}fps)")
        B, C, H, T = blended_latent.shape
        lat_flat = blended_latent.reshape(B, C * H, T)
        lat_resampled = F.interpolate(lat_flat, size=target_latent_frames, mode='nearest-exact')
        blended_latent = lat_resampled.reshape(B, C, H, target_latent_frames)
        print(f"   ✅ Resampled latent shape: {blended_latent.shape}")

    # Decode the blended latent ONCE
    print(f"\n🔊 Decoding blended latent...")

    # Calculate audio length from total duration
    audio_len = int(total_duration_seconds * sr_out)
    print(f"   Total duration: {total_duration_seconds:.2f}s, audio_len: {audio_len} samples")

    p = next(model.dcae.parameters(), None)
    dev = p.device if p is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = p.dtype if p is not None else torch.float32

    x_for_dcae = blended_latent[:1].to(device=dev, dtype=dtype)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dev)

    print(f"   Latent shape to DCAE: {x_for_dcae.shape}")
    print(f"   Expected audio samples: {audio_len}")

    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("   Using overlap decoder")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            result = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
        if isinstance(result, tuple) and len(result) == 2:
            sr_pred, wav_pred = result
        else:
            sr_pred = sr_out
            wav_pred = result
    else:
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    # Handle list or tensor format
    if isinstance(wav_pred, list):
        wav = wav_pred[0].float().cpu() if torch.is_tensor(wav_pred[0]) else wav_pred[0]
    else:
        wav = wav_pred[0].float().cpu()

    # Ensure stereo
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Save output
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_sliding_window_latent_blend_seed{seed}.wav"

    torchaudio.save(str(out_path), wav, sr_pred)

    # Cleanup temp directory
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"\n✅ Latent-space blending complete: {out_path}")
    print(f"   Duration: {wav.shape[-1] / sr_pred:.2f}s")
    print(f"{'='*80}\n")

    return str(out_path)


# ------------------------------------------------------------------------------
# Autoregressive Sliding Window Generation
# ------------------------------------------------------------------------------

def generate_sliding_window_autoregressive(
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
    sr_out: int = 44100,
    use_overlap_decoder: bool = True,
    continuation_weight: float = 0.7,  # How much to weight previous latent vs fresh noise
    **generate_kwargs
) -> str:
    """
    Generate audio using AUTOREGRESSIVE sliding window.

    Key improvement: Each window's initial latent is seeded from the END of
    the previous window's latent (in the overlap region). This provides
    temporal continuity and consistent timbre across windows.

    Args:
        continuation_weight: 0.0 = fresh noise, 1.0 = pure continuation from previous
                            Default 0.7 = 70% previous latent, 30% noise
        ... (other args same as generate_sliding_window_latent_blend)

    Returns:
        Path to generated audio file
    """
    window_frames = int(window_seconds * fps)
    T = piano_roll.shape[-1]
    total_duration = T / fps
    hop_frames = int(window_frames * (1 - overlap_ratio))
    overlap_frames = window_frames - hop_frames

    print(f"\n{'='*80}")
    print(f"AUTOREGRESSIVE SLIDING WINDOW GENERATION")
    print(f"{'='*80}")
    print(f"Total duration: {total_duration:.2f}s ({T} frames)")
    print(f"Window size: {window_seconds:.1f}s ({window_frames} frames)")
    print(f"Overlap: {overlap_ratio*100:.0f}% ({overlap_frames} frames)")
    print(f"Continuation weight: {continuation_weight:.1f} (previous latent vs noise)")

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
    temp_dir = tempfile.mkdtemp(prefix="sliding_window_ar_")

    # Generate windows autoregressively
    latent_segments = []
    previous_latent = None  # Will store end of previous window's latent

    seed = generate_kwargs.get('seed', -1)
    if seed <= 0:
        seed = torch.seed() % 2**31

    for i, window in enumerate(windows):
        print(f"\n   Window {i+1}/{len(windows)}: frames {window['start_frame']}-{window['end_frame']}")

        # Debug: show piano roll activity for this window
        pr_active = np.sum(window['piano_roll'] > 0.1)
        pr_max = np.max(window['piano_roll'])
        amp_sum = np.sum(window['amp'])
        print(f"      🎹 Piano roll: {pr_active} active cells, max={pr_max:.3f}, amp_sum={amp_sum:.3f}")

        window_seed = seed  # Same seed for consistency
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

        # For autoregressive: inject previous latent into initial state
        # This requires modifying how we call generate - we'll pass a custom
        # initial_latent parameter that mixes previous with noise
        window_kwargs = generate_kwargs.copy()
        window_kwargs['seed'] = window_seed
        window_kwargs['original_audio_length'] = original_audio_length
        window_kwargs['target_audio_duration'] = window_duration
        window_kwargs['return_latents'] = True
        window_kwargs['sr_out'] = sr_out
        if window_audio_file:
            window_kwargs['audio_file'] = window_audio_file

        # Generate this window
        latent = generate_fn(
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

        # If we have previous latent, blend the overlap region
        if previous_latent is not None and overlap_frames > 0:
            # The start of this window should match the end of previous window
            # Blend in latent space for the overlap region
            blend_region = min(overlap_frames, latent.shape[-1], previous_latent.shape[-1])

            # Create blending weights (smooth transition)
            blend_weights = torch.linspace(continuation_weight, 0, blend_region, device=latent.device, dtype=latent.dtype)
            blend_weights = blend_weights.view(1, 1, 1, -1)

            # Blend the overlap region
            prev_end = previous_latent[..., -blend_region:]
            curr_start = latent[..., :blend_region]
            blended = blend_weights * prev_end + (1 - blend_weights) * curr_start

            # Replace start of current latent with blended version
            latent = torch.cat([blended, latent[..., blend_region:]], dim=-1)
            print(f"      Blended {blend_region} frames with previous window")

        # Store this latent's end for next window
        previous_latent = latent.clone()

        # Trim to actual length
        expected_latent_frames = window['actual_length']
        if latent.shape[-1] > expected_latent_frames:
            latent = latent[..., :expected_latent_frames]

        latent_segments.append(latent)
        print(f"   Window {i+1} latent: {latent.shape}")

    # Blend latents in latent space
    print(f"\n🔧 Blending {len(latent_segments)} latents in LATENT SPACE...")
    blended_latent = crossfade_blend_latents(latent_segments, windows)
    print(f"   Blended latent shape: {blended_latent.shape}")

    # CRITICAL FIX: Blended latent is at conditioning fps (43.066), but DCAE expects
    # latent at its native resolution (~10.77 fps = 44100/4096). Downsample before decode.
    DCAE_HOP = 4096
    DCAE_SR = 44100
    DCAE_FPS = DCAE_SR / DCAE_HOP  # ~10.766 fps

    total_cond_frames = windows[-1]['start_frame'] + windows[-1]['actual_length']
    total_duration_seconds = total_cond_frames / fps
    target_latent_frames = int(round(total_duration_seconds * DCAE_FPS))

    if blended_latent.shape[-1] != target_latent_frames:
        print(f"   🔄 Resampling latent for DCAE: {blended_latent.shape[-1]} → {target_latent_frames} frames")
        print(f"      (conditioning at {fps:.1f}fps → DCAE at {DCAE_FPS:.2f}fps)")
        B, C, H, T = blended_latent.shape
        lat_flat = blended_latent.reshape(B, C * H, T)
        lat_resampled = F.interpolate(lat_flat, size=target_latent_frames, mode='nearest-exact')
        blended_latent = lat_resampled.reshape(B, C, H, target_latent_frames)
        print(f"   ✅ Resampled latent shape: {blended_latent.shape}")

    # Decode the blended latent ONCE
    print(f"\n🔊 Decoding blended latent...")

    # Calculate audio length from total duration
    audio_len = int(total_duration_seconds * sr_out)
    print(f"   Total duration: {total_duration_seconds:.2f}s, audio_len: {audio_len} samples")

    p = next(model.dcae.parameters(), None)
    dev = p.device if p is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = p.dtype if p is not None else torch.float32

    x_for_dcae = blended_latent[:1].to(device=dev, dtype=dtype)
    audio_lengths = torch.tensor([audio_len], dtype=torch.long, device=dev)

    print(f"   Latent shape to DCAE: {x_for_dcae.shape}")
    print(f"   Expected audio samples: {audio_len}")

    if use_overlap_decoder and hasattr(model.dcae, 'decode_overlap'):
        print("   Using overlap decoder")
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(dev.type=="cuda")):
            result = model.dcae.decode_overlap(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)
        if isinstance(result, tuple) and len(result) == 2:
            sr_pred, wav_pred = result
        else:
            sr_pred = sr_out
            wav_pred = result
    else:
        sr_pred, wav_pred = model.dcae.decode(x_for_dcae, audio_lengths=audio_lengths, sr=sr_out)

    # Handle list or tensor format
    if isinstance(wav_pred, list):
        wav = wav_pred[0].float().cpu() if torch.is_tensor(wav_pred[0]) else wav_pred[0]
    else:
        wav = wav_pred[0].float().cpu()

    if wav.ndim == 1:
        wav = wav.unsqueeze(0)

    # Save output
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{time.strftime('%Y%m%d-%H%M%S')}_sliding_window_ar_seed{seed}.wav"

    torchaudio.save(str(out_path), wav, sr_pred)

    # Cleanup
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)

    print(f"\n✅ Autoregressive sliding window complete: {out_path}")
    print(f"   Duration: {wav.shape[-1] / sr_pred:.2f}s")
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
